# Generated by Django 4.2.17 on 2024-12-13 02:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('App', '0003_predictionresult_delete_history'),
    ]

    operations = [
        migrations.CreateModel(
            name='History',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image_path', models.CharField(max_length=255)),
                ('predicted_label', models.CharField(max_length=255)),
                ('benefits', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.DeleteModel(
            name='PredictionResult',
        ),
    ]
