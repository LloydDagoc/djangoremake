# Generated by Django 4.2.17 on 2024-12-15 09:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('App', '0004_history_delete_predictionresult'),
    ]

    operations = [
        migrations.CreateModel(
            name='PredictionResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('predicted_label', models.CharField(max_length=255)),
                ('benefits', models.TextField()),
                ('confidence', models.FloatField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.DeleteModel(
            name='History',
        ),
    ]
