from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0006_chatbotinteraction"),
    ]

    operations = [
        migrations.AddField(
            model_name="doctor",
            name="profile_image",
            field=models.ImageField(blank=True, null=True, upload_to="doctors/profiles/"),
        ),
        migrations.AddField(
            model_name="doctornote",
            name="attachment",
            field=models.FileField(blank=True, null=True, upload_to="doctor_notes/attachments/"),
        ),
        migrations.AddField(
            model_name="healthmetrics",
            name="source_upload",
            field=models.FileField(blank=True, null=True, upload_to="healthmetrics/uploads/"),
        ),
        migrations.AddField(
            model_name="patient",
            name="profile_image",
            field=models.ImageField(blank=True, null=True, upload_to="patients/profiles/"),
        ),
        migrations.AddField(
            model_name="prediction",
            name="report_file",
            field=models.FileField(blank=True, null=True, upload_to="predictions/reports/"),
        ),
    ]
