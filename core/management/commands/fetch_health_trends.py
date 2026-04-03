from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db import OperationalError, ProgrammingError, connection
from django.db import transaction
from django.utils import timezone

from core.health_trends import analyze_headline, fetch_health_headlines
from core.models import HealthTrend
from core.offline_utils import is_online


class Command(BaseCommand):
    help = "Fetch and process trending health headlines into HealthTrend records"

    def add_arguments(self, parser):
        parser.add_argument(
            "--limit",
            type=int,
            default=10,
            help="Maximum number of headlines to fetch (default: 10)",
        )

    def handle(self, *args, **options):
        if not self._healthtrend_table_available():
            self.stdout.write(
                self.style.WARNING(
                    "HealthTrend table not found in current database. Skipping trend fetch in this mode."
                )
            )
            return

        limit = max(5, min(int(options.get("limit", 10)), 20))
        now = timezone.now()
        keep_since = now - timedelta(days=7)

        # Keep page rendering offline-safe by using local records when internet is unavailable.
        if not is_online():
            self.stdout.write(
                self.style.WARNING(
                    "Internet appears unavailable. Skipping live fetch and keeping existing local trend data."
                )
            )
            stale_deleted, _ = HealthTrend.objects.filter(created_at__lt=keep_since).delete()
            self.stdout.write(self.style.SUCCESS(f"Removed stale trends older than 7 days: {stale_deleted}"))
            return

        headlines = fetch_health_headlines(limit=limit)
        if not headlines:
            self.stdout.write(self.style.WARNING("No headlines were fetched from configured sources."))
            stale_deleted, _ = HealthTrend.objects.filter(created_at__lt=keep_since).delete()
            self.stdout.write(self.style.SUCCESS(f"Removed stale trends older than 7 days: {stale_deleted}"))
            return

        created_count = 0
        updated_count = 0

        with transaction.atomic():
            # Mark previous entries as non-local; this fetch becomes the latest local dataset snapshot.
            HealthTrend.objects.update(is_local=False)
            seen_diseases = set()

            for item in headlines:
                title = (item.get("title") or "").strip()
                source_url = (item.get("source_url") or "").strip() or "https://news.google.com/"
                if not title:
                    continue

                extracted = analyze_headline(title, source_url=source_url)
                disease_name = extracted["disease_name"]
                if disease_name in seen_diseases:
                    continue

                trend, created = HealthTrend.objects.get_or_create(
                    title=title,
                    source_url=source_url,
                    defaults={
                        "disease_name": disease_name,
                        "risk_level": extracted["risk_level"],
                        "confidence_score": extracted["confidence_score"],
                        "description": extracted["description"],
                        "preventive_advice": extracted["preventive_advice"],
                        "is_global": True,
                        "is_local": True,
                    },
                )

                seen_diseases.add(disease_name)

                if created:
                    created_count += 1
                    continue

                trend.disease_name = disease_name
                trend.risk_level = extracted["risk_level"]
                trend.confidence_score = extracted["confidence_score"]
                trend.description = extracted["description"]
                trend.preventive_advice = extracted["preventive_advice"]
                trend.is_global = True
                trend.is_local = True
                trend.save(
                    update_fields=[
                        "disease_name",
                        "risk_level",
                        "confidence_score",
                        "description",
                        "preventive_advice",
                        "is_global",
                        "is_local",
                    ]
                )
                updated_count += 1

            stale_deleted, _ = HealthTrend.objects.filter(created_at__lt=keep_since).delete()

        self.stdout.write(self.style.SUCCESS("Trending health insights fetch completed."))
        self.stdout.write(f"Created: {created_count}")
        self.stdout.write(f"Updated: {updated_count}")
        self.stdout.write(f"Total fetched headlines: {len(headlines)}")
        self.stdout.write(f"Removed stale trends older than 7 days: {stale_deleted}")

    def _healthtrend_table_available(self) -> bool:
        try:
            table_names = connection.introspection.table_names()
            return HealthTrend._meta.db_table in table_names
        except (OperationalError, ProgrammingError):
            return False
