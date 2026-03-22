from django.db import models

class Doctor(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    specialization = models.CharField(max_length=100)
    experience_years = models.PositiveIntegerField(default=0)
    hospital = models.CharField(max_length=100)
    password = models.CharField(max_length=200)
    profile_image = models.ImageField(upload_to="doctors/profiles/", null=True, blank=True)
    # Offline-first: Track whether this record has been synced to cloud
    is_synced = models.BooleanField(default=True, help_text="Whether this record is synced with cloud database")
    synced_at = models.DateTimeField(null=True, blank=True, help_text="Timestamp of last successful cloud sync")

    def __str__(self):
        return self.name


class Patient(models.Model):
    GENDER_CHOICES = [
        ("Male", "Male"),
        ("Female", "Female"),
        ("Other", "Other"),
    ]

    name = models.CharField(max_length=100)
    age = models.PositiveIntegerField()
    gender = models.CharField(max_length=20, choices=GENDER_CHOICES)
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, related_name="patients")
    assigned_doctor = models.ForeignKey(
        Doctor,
        on_delete=models.SET_NULL,
        related_name="auto_assigned_patients",
        null=True,
        blank=True,
    )
    email = models.EmailField(unique=True)
    mobile = models.CharField(max_length=20)
    username = models.CharField(max_length=150, unique=True)
    password = models.CharField(max_length=200)
    profile_image = models.ImageField(upload_to="patients/profiles/", null=True, blank=True)
    suggested_disease = models.CharField(max_length=100, blank=True)
    disease_risk_level = models.CharField(max_length=10, blank=True)
    suggested_tests = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # Offline-first: Track whether this record has been synced to cloud
    is_synced = models.BooleanField(default=True, help_text="Whether this record is synced with cloud database")
    synced_at = models.DateTimeField(null=True, blank=True, help_text="Timestamp of last successful cloud sync")

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class HealthMetrics(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="health_metrics")
    date = models.DateField()
    steps = models.IntegerField(default=0)
    sleep_hours = models.FloatField(null=True, blank=True)
    sleep_quality = models.FloatField(null=True, blank=True)
    heart_rate = models.FloatField(null=True, blank=True)
    stress_level = models.FloatField(null=True, blank=True)
    blood_oxygen = models.FloatField(null=True, blank=True)
    activity_level = models.CharField(max_length=20, blank=True)
    calories_burned = models.FloatField(default=0)
    data_source = models.CharField(max_length=30, default="Manual entry")
    source_upload = models.FileField(upload_to="healthmetrics/uploads/", null=True, blank=True)
    # Offline-first: Track whether this record has been synced to cloud
    is_synced = models.BooleanField(default=True, help_text="Whether this record is synced with cloud database")
    synced_at = models.DateTimeField(null=True, blank=True, help_text="Timestamp of last successful cloud sync")

    class Meta:
        ordering = ["-date"]
        constraints = [
            models.UniqueConstraint(fields=["patient", "date"], name="unique_patient_metric_date"),
        ]

    def __str__(self):
        return f"{self.patient.name} - {self.date}"


class Prediction(models.Model):
    PRIORITY_HIGH = "High"
    PRIORITY_MEDIUM = "Medium"
    PRIORITY_LOW = "Low"
    PRIORITY_CHOICES = [
        (PRIORITY_HIGH, "High"),
        (PRIORITY_MEDIUM, "Medium"),
        (PRIORITY_LOW, "Low"),
    ]

    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="predictions")
    risk_score = models.FloatField()
    priority = models.CharField(max_length=10, choices=PRIORITY_CHOICES)
    explanation_text = models.TextField()
    predicted_label = models.CharField(max_length=50, blank=True)
    data_source = models.CharField(max_length=30, default="Manual entry")
    risk_factors = models.TextField(blank=True)
    readable_ai_explanation = models.TextField(blank=True)
    nervous_system_recovery_score = models.FloatField(null=True, blank=True)
    nervous_system_explanation = models.TextField(blank=True)
    stress_sleep_insight = models.TextField(blank=True)
    recommended_follow_up = models.TextField(blank=True)
    lifestyle_recommendations = models.TextField(blank=True)
    report_file = models.FileField(upload_to="predictions/reports/", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # Offline-first: Track whether this record has been synced to cloud
    is_synced = models.BooleanField(default=True, help_text="Whether this record is synced with cloud database")
    synced_at = models.DateTimeField(null=True, blank=True, help_text="Timestamp of last successful cloud sync")

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.patient.name} - {self.risk_score:.1f}"


class ChatbotInteraction(models.Model):
    ROLE_PATIENT = "patient"
    ROLE_DOCTOR = "doctor"
    ROLE_SYSTEM = "system"
    ROLE_CHOICES = [
        (ROLE_PATIENT, "Patient"),
        (ROLE_DOCTOR, "Doctor"),
        (ROLE_SYSTEM, "System"),
    ]

    patient = models.ForeignKey(Patient, on_delete=models.SET_NULL, null=True, blank=True, related_name="chatbot_interactions")
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_PATIENT)
    intent = models.CharField(max_length=50, blank=True)
    question = models.TextField()
    response = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # Offline-first: Track whether this record has been synced to cloud
    is_synced = models.BooleanField(default=True, help_text="Whether this record is synced with cloud database")
    synced_at = models.DateTimeField(null=True, blank=True, help_text="Timestamp of last successful cloud sync")

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.role} interaction @ {self.created_at:%Y-%m-%d %H:%M}"


class DoctorNote(models.Model):
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, related_name="notes")
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="doctor_notes")
    note = models.TextField()
    attachment = models.FileField(upload_to="doctor_notes/attachments/", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # Offline-first: Track whether this record has been synced to cloud
    is_synced = models.BooleanField(default=True, help_text="Whether this record is synced with cloud database")
    synced_at = models.DateTimeField(null=True, blank=True, help_text="Timestamp of last successful cloud sync")

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.doctor.name} -> {self.patient.name}"