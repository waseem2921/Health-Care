from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0008_offline_first_sync_fields"),
    ]

    operations = [
        migrations.CreateModel(
            name="HealthTrend",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("title", models.CharField(max_length=255)),
                ("disease_name", models.CharField(max_length=120)),
                (
                    "risk_level",
                    models.CharField(
                        choices=[("Low", "Low"), ("Moderate", "Moderate"), ("High", "High")],
                        default="Low",
                        max_length=10,
                    ),
                ),
                ("description", models.TextField()),
                ("preventive_advice", models.TextField()),
                ("source_url", models.URLField(max_length=500)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("is_global", models.BooleanField(default=True)),
                ("is_local", models.BooleanField(default=True)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.AddIndex(
            model_name="healthtrend",
            index=models.Index(fields=["-created_at"], name="core_health_created_feee92_idx"),
        ),
        migrations.AddIndex(
            model_name="healthtrend",
            index=models.Index(fields=["risk_level"], name="core_health_risk_le_63266c_idx"),
        ),
        migrations.AddIndex(
            model_name="healthtrend",
            index=models.Index(fields=["disease_name"], name="core_health_disease_19b4e6_idx"),
        ),
        migrations.AddConstraint(
            model_name="healthtrend",
            constraint=models.UniqueConstraint(fields=("title", "source_url"), name="unique_trend_title_source"),
        ),
    ]
