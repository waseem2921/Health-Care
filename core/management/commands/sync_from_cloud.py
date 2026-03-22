"""
Django management command to sync records from cloud database (NeonDB) to local SQLite.

Usage:
    python manage.py sync_from_cloud [--dry-run] [--model=MODEL_NAME] [--batch-size=200]

Options:
    --dry-run: Show what would be synced without making changes
    --model: Only sync a specific model (e.g., Patient, HealthMetrics)
    --batch-size: Number of records processed per batch (default: 200)
"""

import os
import re
from typing import Any, Dict, Mapping, Optional, Tuple

import dj_database_url
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import DatabaseError, OperationalError, connections, transaction
from django.utils import timezone

from core.models import (
    ChatbotInteraction,
    Doctor,
    DoctorNote,
    HealthMetrics,
    Patient,
    Prediction,
)

# Respect FK dependency order while importing from cloud.
SYNCABLE_MODELS = {
    "Doctor": Doctor,
    "Patient": Patient,
    "HealthMetrics": HealthMetrics,
    "Prediction": Prediction,
    "ChatbotInteraction": ChatbotInteraction,
    "DoctorNote": DoctorNote,
}


class Command(BaseCommand):
    help = "Sync records from cloud database (NeonDB) to local SQLite"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be synced without making changes",
        )
        parser.add_argument(
            "--model",
            type=str,
            help="Only sync a specific model (e.g., Patient, HealthMetrics)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=200,
            help="Batch size for iterating cloud records (default: 200)",
        )

    def handle(self, *args, **options):
        dry_run = options.get("dry_run", False)
        model_name = options.get("model")
        batch_size = max(1, int(options.get("batch_size", 200)))

        self.stdout.write(self.style.SUCCESS("⬇️  Starting cloud-to-local sync...\n"))

        models_to_sync = self._get_models_to_sync(model_name)
        if not models_to_sync:
            raise CommandError(f"Invalid model name: {model_name}")

        cloud_alias, local_alias = self._prepare_connections()

        try:
            total_seen = 0
            total_created = 0
            total_updated = 0

            self.stdout.write(
                f"Source: CLOUD ({cloud_alias}) -> Target: LOCAL SQLITE ({local_alias})"
            )
            self.stdout.write(f"Models: {', '.join(models_to_sync.keys())}\n")

            for model_label, model_class in models_to_sync.items():
                seen, created, updated = self._sync_model(
                    model_label=model_label,
                    model_class=model_class,
                    cloud_alias=cloud_alias,
                    local_alias=local_alias,
                    dry_run=dry_run,
                    batch_size=batch_size,
                )
                total_seen += seen
                total_created += created
                total_updated += updated

            self.stdout.write("\n" + "=" * 60)
            if dry_run:
                self.stdout.write(self.style.SUCCESS("📋 DRY RUN SUMMARY"))
            else:
                self.stdout.write(self.style.SUCCESS("✅ SYNC SUMMARY"))
            self.stdout.write(f"   Records scanned: {total_seen}")
            self.stdout.write(f"   Records created locally: {total_created}")
            self.stdout.write(f"   Records updated locally: {total_updated}")
            self.stdout.write("=" * 60 + "\n")
        finally:
            self._cleanup_connections(cloud_alias, local_alias)

    def _get_models_to_sync(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        if model_name:
            for key in SYNCABLE_MODELS:
                if key.lower() == model_name.lower():
                    return {key: SYNCABLE_MODELS[key]}
            return {}
        return SYNCABLE_MODELS

    def _prepare_connections(self) -> Tuple[str, str]:
        cloud_alias = "sync_cloud"
        local_alias = "sync_local"

        raw_database_url = os.getenv("DATABASE_URL", "").strip()
        database_url = raw_database_url.strip('"').strip("'")
        embedded_match = re.search(r"(postgres(?:ql)?://[^\s'\"]+)", database_url)
        if embedded_match:
            database_url = embedded_match.group(1)

        database_url_valid = (
            bool(database_url)
            and database_url not in {"://", "postgresql://", "postgres://"}
            and "://" in database_url
        )

        if not database_url_valid:
            raise CommandError("DATABASE_URL is missing or malformed. Cannot pull from cloud.")

        local_db_path = str(getattr(settings, "LOCAL_DB_PATH", settings.BASE_DIR / "local_offline.db"))

        cloud_db = dj_database_url.parse(
            database_url,
            conn_max_age=60,
            ssl_require=True,
        )
        local_db = {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": local_db_path,
        }

        settings.DATABASES[cloud_alias] = self._normalize_database_config(cloud_db)
        settings.DATABASES[local_alias] = self._normalize_database_config(local_db)

        try:
            connections[cloud_alias].ensure_connection()
        except Exception as exc:
            raise CommandError(f"Cloud database unreachable: {exc}") from exc

        try:
            connections[local_alias].ensure_connection()
        except Exception as exc:
            raise CommandError(f"Local SQLite unavailable: {exc}") from exc

        return cloud_alias, local_alias

    def _normalize_database_config(self, db_config: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = {
            "ATOMIC_REQUESTS": False,
            "AUTOCOMMIT": True,
            "CONN_HEALTH_CHECKS": False,
            "CONN_MAX_AGE": 0,
            "ENGINE": "",
            "HOST": "",
            "NAME": "",
            "OPTIONS": {},
            "PASSWORD": "",
            "PORT": "",
            "TIME_ZONE": None,
            "USER": "",
            "TEST": {
                "CHARSET": None,
                "COLLATION": None,
                "MIGRATE": True,
                "MIRROR": None,
                "NAME": None,
            },
        }
        normalized.update(db_config or {})
        if "OPTIONS" not in normalized or normalized["OPTIONS"] is None:
            normalized["OPTIONS"] = {}
        if "TEST" not in normalized or normalized["TEST"] is None:
            normalized["TEST"] = {
                "CHARSET": None,
                "COLLATION": None,
                "MIGRATE": True,
                "MIRROR": None,
                "NAME": None,
            }
        return normalized

    def _cleanup_connections(self, cloud_alias: str, local_alias: str):
        for alias in (cloud_alias, local_alias):
            try:
                if alias in connections:
                    connections[alias].close()
            except Exception:
                pass

    def _sync_model(
        self,
        model_label: str,
        model_class,
        cloud_alias: str,
        local_alias: str,
        dry_run: bool,
        batch_size: int,
    ) -> Tuple[int, int, int]:
        self.stdout.write(f"📦 Syncing {model_label}...")

        source_qs = model_class.objects.using(cloud_alias).all().order_by("pk")
        total = source_qs.count()

        if total == 0:
            self.stdout.write(f"   ✓ No cloud records found for {model_label}")
            return 0, 0, 0

        if dry_run:
            self.stdout.write(f"   Would scan {total} record(s)")
            return total, 0, 0

        created = 0
        updated = 0
        pk_name = model_class._meta.pk.name

        with transaction.atomic(using=local_alias):
            for source_obj in source_qs.iterator(chunk_size=batch_size):
                defaults = self._build_defaults(model_class, source_obj)
                pk_value = getattr(source_obj, pk_name)
                _, was_created = model_class.objects.using(local_alias).update_or_create(
                    **{pk_name: pk_value},
                    defaults=defaults,
                )
                if was_created:
                    created += 1
                else:
                    updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"   ✓ {model_label}: scanned={total}, created={created}, updated={updated}"
            )
        )
        return total, created, updated

    def _build_defaults(self, model_class, source_obj) -> Dict:
        defaults = {}
        for field in model_class._meta.concrete_fields:
            if field.primary_key:
                continue
            defaults[field.attname] = getattr(source_obj, field.attname)

        # Cloud record pulled locally should be marked synced.
        if "is_synced" in defaults:
            defaults["is_synced"] = True
        if "synced_at" in defaults:
            defaults["synced_at"] = timezone.now()

        return defaults
