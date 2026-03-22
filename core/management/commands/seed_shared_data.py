from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from core.models import Doctor, HealthMetrics, Patient, Prediction


class Command(BaseCommand):
    help = "Seed shared starter data (doctors, patients, health metrics, predictions)."

    @transaction.atomic
    def handle(self, *args, **options):
        doctors_payload = [
            {
                "name": "Dr Ahmed",
                "email": "ahmed@hospital.com",
                "specialization": "Cardiologist",
                "experience_years": 10,
                "hospital": "Apollo Hospital",
                "password": "1234",
            },
            {
                "name": "Dr Priya Sharma",
                "email": "priya@hospital.com",
                "specialization": "Neurologist",
                "experience_years": 8,
                "hospital": "Care Hospitals",
                "password": "1234",
            },
            {
                "name": "Dr Ramesh Kumar",
                "email": "ramesh@hospital.com",
                "specialization": "General Physician",
                "experience_years": 11,
                "hospital": "City Medical Center",
                "password": "1234",
            },
            {
                "name": "Dr Fatima Khan",
                "email": "fatima@hospital.com",
                "specialization": "Endocrinologist",
                "experience_years": 9,
                "hospital": "Sunrise Clinic",
                "password": "1234",
            },
            {
                "name": "Dr David Wilson",
                "email": "david@hospital.com",
                "specialization": "Orthopedic",
                "experience_years": 12,
                "hospital": "Global Hospital",
                "password": "1234",
            },
        ]

        created_doctors = []
        for doctor_data in doctors_payload:
            doctor, _ = Doctor.objects.update_or_create(
                email=doctor_data["email"],
                defaults=doctor_data,
            )
            created_doctors.append(doctor)

        patients_payload = [
            {
                "name": "Aisha Rahman",
                "age": 33,
                "gender": "Female",
                "email": "aisha.rahman@example.com",
                "mobile": "9000000001",
                "username": "aisha",
                "password": "Patient@123",
                "doctor": created_doctors[0],
            },
            {
                "name": "Arjun Mehta",
                "age": 41,
                "gender": "Male",
                "email": "arjun.mehta@example.com",
                "mobile": "9000000002",
                "username": "arjun",
                "password": "Patient@123",
                "doctor": created_doctors[1],
            },
            {
                "name": "Mina Das",
                "age": 29,
                "gender": "Female",
                "email": "mina.das@example.com",
                "mobile": "9000000003",
                "username": "mina",
                "password": "Patient@123",
                "doctor": created_doctors[2],
            },
        ]

        created_patients = []
        for patient_data in patients_payload:
            primary_doctor = patient_data["doctor"]
            patient, _ = Patient.objects.update_or_create(
                username=patient_data["username"],
                defaults={
                    **patient_data,
                    "doctor": primary_doctor,
                    "assigned_doctor": primary_doctor,
                    "suggested_disease": "Lifestyle Imbalance",
                    "disease_risk_level": "Moderate",
                    "suggested_tests": "Lifestyle Assessment|Sleep Study",
                },
            )
            created_patients.append(patient)

        today = timezone.localdate()
        metrics_payload = [
            {
                "steps": 5200,
                "sleep_hours": 6.2,
                "sleep_quality": 5.8,
                "heart_rate": 83,
                "stress_level": 6.4,
                "blood_oxygen": 96.3,
                "activity_level": "Moderate",
                "calories_burned": 1850,
            },
            {
                "steps": 7900,
                "sleep_hours": 7.1,
                "sleep_quality": 7.2,
                "heart_rate": 74,
                "stress_level": 4.1,
                "blood_oxygen": 97.1,
                "activity_level": "High",
                "calories_burned": 2180,
            },
            {
                "steps": 4300,
                "sleep_hours": 5.7,
                "sleep_quality": 5.0,
                "heart_rate": 89,
                "stress_level": 7.1,
                "blood_oxygen": 95.2,
                "activity_level": "Low",
                "calories_burned": 1710,
            },
        ]

        prediction_payload = [
            {"risk_score": 68.4, "priority": Prediction.PRIORITY_MEDIUM},
            {"risk_score": 32.8, "priority": Prediction.PRIORITY_LOW},
            {"risk_score": 79.3, "priority": Prediction.PRIORITY_HIGH},
        ]

        for patient, metric_data, pred_data in zip(created_patients, metrics_payload, prediction_payload):
            HealthMetrics.objects.update_or_create(
                patient=patient,
                date=today,
                defaults={
                    **metric_data,
                    "data_source": "Seed data",
                },
            )

            prediction_defaults = {
                "risk_score": pred_data["risk_score"],
                "priority": pred_data["priority"],
                "explanation_text": "Seeded prediction for shared dashboard validation.",
                "predicted_label": "Moderate Risk",
                "data_source": "Seed data",
                "risk_factors": "stress|sleep|activity",
                "readable_ai_explanation": "Shared seeded AI explanation.",
                "nervous_system_recovery_score": 61.5,
                "nervous_system_explanation": "Moderate recovery load detected.",
                "stress_sleep_insight": "Stress and sleep are moderately correlated.",
                "recommended_follow_up": "Lifestyle review|Sleep hygiene",
                "lifestyle_recommendations": "Increase activity|Improve sleep",
            }

            existing_prediction = Prediction.objects.filter(
                patient=patient,
                created_at__date=today,
            ).order_by("-created_at").first()

            if existing_prediction is not None:
                for key, value in prediction_defaults.items():
                    setattr(existing_prediction, key, value)
                existing_prediction.save(update_fields=list(prediction_defaults.keys()))
            else:
                Prediction.objects.create(patient=patient, **prediction_defaults)

        self.stdout.write(self.style.SUCCESS("Seed data ready: doctors, patients, metrics, and predictions synced."))
