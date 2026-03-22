from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.management.base import BaseCommand
from django.db import connection

from core.models import Doctor, DoctorNote, HealthMetrics, Patient, Prediction


class Command(BaseCommand):
    help = "Verify NeonDB and Cloudinary integration with runtime checks."

    def handle(self, *args, **options):
        self.stdout.write("Running NeonDB + Cloudinary checks...")

        # 1) Database connectivity check
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            self.stdout.write(self.style.SUCCESS("Database connectivity: OK"))
        except Exception as exc:
            self.stdout.write(self.style.ERROR(f"Database connectivity failed: {exc}"))
            return

        db_cfg = settings.DATABASES.get("default", {})
        self.stdout.write(
            f"DB Engine: {db_cfg.get('ENGINE', 'unknown')} | DB Name: {db_cfg.get('NAME', 'unknown')}"
        )

        media_backend = settings.STORAGES.get("default", {}).get("BACKEND", "unknown")
        self.stdout.write(f"Media backend: {media_backend}")
        cloudinary_active = media_backend == "cloudinary_storage.storage.MediaCloudinaryStorage"
        if cloudinary_active:
            self.stdout.write(self.style.SUCCESS("Cloudinary backend is active."))
        else:
            self.stdout.write(
                self.style.WARNING(
                    "Cloudinary backend is not active. Set CLOUDINARY_* env vars (and optionally REQUIRE_CLOUDINARY=true)."
                )
            )

        # 2) Shared data visibility checks
        self.stdout.write(
            "Counts => "
            f"Doctors: {Doctor.objects.count()}, "
            f"Patients: {Patient.objects.count()}, "
            f"HealthMetrics: {HealthMetrics.objects.count()}, "
            f"Predictions: {Prediction.objects.count()}, "
            f"DoctorNotes: {DoctorNote.objects.count()}"
        )

        # 3) Cloudinary/default storage upload check
        try:
            check_path = default_storage.save(
                "healthchecks/integration_check.txt",
                ContentFile(b"NeonDB + Cloudinary integration check"),
            )
            file_url = default_storage.url(check_path)
            if cloudinary_active:
                self.stdout.write(self.style.SUCCESS("Cloudinary upload: OK"))
            else:
                self.stdout.write(self.style.WARNING("Media upload succeeded using local fallback storage."))
            self.stdout.write(f"Cloud file URL: {file_url}")
        except Exception as exc:
            self.stdout.write(self.style.ERROR(f"Cloud storage check failed: {exc}"))

        self.stdout.write(self.style.SUCCESS("Verification completed."))
