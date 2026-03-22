from django.contrib import admin
from .models import ChatbotInteraction, Doctor, DoctorNote, HealthMetrics, Patient, Prediction

admin.site.register(Doctor)
admin.site.register(Patient)
admin.site.register(HealthMetrics)
admin.site.register(Prediction)
admin.site.register(DoctorNote)
admin.site.register(ChatbotInteraction)

