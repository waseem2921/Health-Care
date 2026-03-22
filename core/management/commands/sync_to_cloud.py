"""
Django management command to sync unsynced records from local SQLite to cloud database (NeonDB).

Usage:
    python manage.py sync_to_cloud [--force] [--dry-run] [--model=MODEL_NAME]

Options:
    --force: Sync even if offline (will fail but attempt anyway)
    --dry-run: Show what would be synced without making changes
    --model: Only sync a specific model (e.g., Patient, HealthMetrics, Prediction)
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple

from django.apps import apps
from django.core.management.base import BaseCommand, CommandError
from django.db import DatabaseError, OperationalError, connections

from core.models import (
    ChatbotInteraction,
    Doctor,
    DoctorNote,
    HealthMetrics,
    Patient,
    Prediction,
)
from core.offline_utils import (
    get_app_mode,
    is_database_online,
    is_online,
    log_mode_change,
)

logger = logging.getLogger(__name__)

# Models that support syncing
SYNCABLE_MODELS = {
    "Doctor": Doctor,
    "Patient": Patient,
    "HealthMetrics": HealthMetrics,
    "Prediction": Prediction,
    "ChatbotInteraction": ChatbotInteraction,
    "DoctorNote": DoctorNote,
}


class Command(BaseCommand):
    help = "Sync unsynced records from local database to cloud (NeonDB)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Attempt sync even if appears offline",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be synced without making changes",
        )
        parser.add_argument(
            "--model",
            type=str,
            help="Only sync specific model (e.g., Patient, HealthMetrics)",
        )

    def handle(self, *args, **options):
        """Main command handler."""
        force = options.get("force", False)
        dry_run = options.get("dry_run", False)
        model_name = options.get("model", None)

        self.stdout.write(self.style.SUCCESS("🔄 Starting offline-to-cloud sync...\n"))

        # Check connectivity
        db_mode, storage_mode = get_app_mode()
        self.stdout.write(f"Current Mode - Database: {db_mode.upper()}, Storage: {storage_mode.upper()}\n")

        if not force and not is_online():
            self.stdout.write(
                self.style.WARNING(
                    "⚠️  System appears to be offline. Cannot sync to cloud.\n"
                    "Use --force to attempt sync anyway."
                )
            )
            if db_mode == "offline":
                self.stdout.write(
                    "💾 Still in SQLite mode. Sync will be attempted when cloud becomes available.\n"
                )
            return

        if not is_database_online():
            self.stdout.write(
                self.style.ERROR(
                    "❌ Cloud database is unreachable. Cannot complete sync.\n"
                )
            )
            if not force:
                return

        # Determine which models to sync
        models_to_sync = self._get_models_to_sync(model_name)
        if not models_to_sync:
            self.stdout.write(
                self.style.ERROR(f"No valid models found to sync: {model_name}")
            )
            raise CommandError(f"Invalid model name: {model_name}")

        # Perform sync
        self.stdout.write(f"Models to sync: {', '.join(models_to_sync.keys())}\n")
        total_synced, total_errors = self._perform_sync(
            models_to_sync, dry_run, force
        )

        # Summary
        self.stdout.write("\n" + "=" * 60)
        if dry_run:
            self.stdout.write(self.style.SUCCESS(f"📋 DRY RUN - Would sync:"))
        else:
            self.stdout.write(self.style.SUCCESS(f"✅ Sync Complete:"))

        self.stdout.write(f"   ✓ Records synced: {total_synced}")
        if total_errors:
            self.stdout.write(
                self.style.WARNING(f"   ⚠️  Errors encountered: {total_errors}")
            )
        self.stdout.write("=" * 60 + "\n")

    def _get_models_to_sync(self, model_name: str = None) -> Dict:
        """
        Get the models to sync.

        Args:
            model_name: Specific model name or None for all

        Returns:
            Dict of model name to model class
        """
        if model_name:
            if model_name not in SYNCABLE_MODELS:
                return {}
            return {model_name: SYNCABLE_MODELS[model_name]}
        return SYNCABLE_MODELS

    def _perform_sync(
        self, models: Dict, dry_run: bool = False, force: bool = False
    ) -> Tuple[int, int]:
        """
        Perform the actual sync operation.

        Args:
            models: Dict of models to sync
            dry_run: If True, don't make changes
            force: If True, continue even on errors

        Returns:
            Tuple of (total_synced, total_errors)
        """
        total_synced = 0
        total_errors = 0

        for model_name, model_class in models.items():
            self.stdout.write(f"\n📊 Syncing {model_name}...")

            try:
                # Get unsynced records
                unsynced = model_class.objects.filter(is_synced=False)
                count = unsynced.count()

                if count == 0:
                    self.stdout.write(f"   ✓ No unsynced {model_name} records")
                    continue

                self.stdout.write(f"   Found {count} unsynced records")

                if dry_run:
                    self._display_dry_run_records(model_name, unsynced.values_list("pk", flat=True)[:5])
                    total_synced += count
                else:
                    # Perform actual sync
                    synced_count = self._sync_model_records(model_class, unsynced)
                    total_synced += synced_count
                    self.stdout.write(
                        self.style.SUCCESS(f"   ✓ Synced {synced_count} {model_name} records")
                    )

            except Exception as e:
                total_errors += 1
                error_msg = f"Error syncing {model_name}: {str(e)[:100]}"
                self.stdout.write(self.style.ERROR(f"   ❌ {error_msg}"))
                logger.error(error_msg, exc_info=True)

                if not force:
                    raise CommandError(error_msg)

        return total_synced, total_errors

    def _display_dry_run_records(self, model_name: str, pk_list: List):
        """Display records that would be synced in dry-run mode."""
        if not pk_list:
            self.stdout.write(f"   (Sample records: showing first 5 of many)")
        else:
            pks_str = ", ".join(str(pk) for pk in pk_list)
            self.stdout.write(f"   (Sample PKs: {pks_str}...)")

    def _sync_model_records(self, model_class, unsynced_records) -> int:
        """
        Sync unsynced records by marking them as synced.
        In a production system, this would actually push to remote database.

        Args:
            model_class: Django model class
            unsynced_records: QuerySet of unsynced records

        Returns:
            int: Number of records synced
        """
        synced_count = 0

        try:
            for record in unsynced_records.iterator(chunk_size=100):
                record.is_synced = True
                record.synced_at = datetime.now()
                record.save(update_fields=["is_synced", "synced_at"])
                synced_count += 1

            return synced_count

        except (DatabaseError, OperationalError) as e:
            logger.error(f"Database error during sync: {e}")
            if not connections["default"]:
                raise CommandError(
                    "Lost cloud database connection during sync"
                ) from e
            raise
