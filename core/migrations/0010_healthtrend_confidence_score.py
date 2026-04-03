from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0009_healthtrend"),
    ]

    operations = [
        migrations.AddField(
            model_name="healthtrend",
            name="confidence_score",
            field=models.FloatField(default=50.0),
        ),
    ]
