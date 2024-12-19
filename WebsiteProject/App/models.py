from django.db import models

class PredictionResult(models.Model):
    label = models.CharField(max_length=255)
    confidence = models.FloatField()
    benefits = models.TextField()

    def __str__(self):
        return f"{self.label} ({self.confidence}%)"
