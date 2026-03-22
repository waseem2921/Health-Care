# Generated migration for offline-first is_synced field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0007_cloudinary_media_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='doctor',
            name='is_synced',
            field=models.BooleanField(default=True, help_text='Whether this record is synced with cloud database'),
        ),
        migrations.AddField(
            model_name='doctor',
            name='synced_at',
            field=models.DateTimeField(blank=True, help_text='Timestamp of last successful cloud sync', null=True),
        ),
        migrations.AddField(
            model_name='patient',
            name='is_synced',
            field=models.BooleanField(default=True, help_text='Whether this record is synced with cloud database'),
        ),
        migrations.AddField(
            model_name='patient',
            name='synced_at',
            field=models.DateTimeField(blank=True, help_text='Timestamp of last successful cloud sync', null=True),
        ),
        migrations.AddField(
            model_name='healthmetrics',
            name='is_synced',
            field=models.BooleanField(default=True, help_text='Whether this record is synced with cloud database'),
        ),
        migrations.AddField(
            model_name='healthmetrics',
            name='synced_at',
            field=models.DateTimeField(blank=True, help_text='Timestamp of last successful cloud sync', null=True),
        ),
        migrations.AddField(
            model_name='prediction',
            name='is_synced',
            field=models.BooleanField(default=True, help_text='Whether this record is synced with cloud database'),
        ),
        migrations.AddField(
            model_name='prediction',
            name='synced_at',
            field=models.DateTimeField(blank=True, help_text='Timestamp of last successful cloud sync', null=True),
        ),
        migrations.AddField(
            model_name='chatbotinteraction',
            name='is_synced',
            field=models.BooleanField(default=True, help_text='Whether this record is synced with cloud database'),
        ),
        migrations.AddField(
            model_name='chatbotinteraction',
            name='synced_at',
            field=models.DateTimeField(blank=True, help_text='Timestamp of last successful cloud sync', null=True),
        ),
        migrations.AddField(
            model_name='doctornote',
            name='is_synced',
            field=models.BooleanField(default=True, help_text='Whether this record is synced with cloud database'),
        ),
        migrations.AddField(
            model_name='doctornote',
            name='synced_at',
            field=models.DateTimeField(blank=True, help_text='Timestamp of last successful cloud sync', null=True),
        ),
    ]
