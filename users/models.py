from django.db import models

# Create your models here.

class UserRegistration(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(blank=False)
    phonenumber = models.IntegerField()
    address = models.TextField()
    username = models.CharField(max_length=20)
    password = models.CharField(max_length=20)

    is_active = models.BooleanField(default=False)  

    def __str__(self):
        return self.name
    
from django.db import models
from django.contrib.auth.models import User

class QuestionnaireResponse(models.Model):
    wake_freshness = models.CharField(max_length=10)
    wake_frequency = models.CharField(max_length=10)
    stress_level = models.CharField(max_length=10)
    stress_management = models.CharField(max_length=50)
    exercise_frequency = models.CharField(max_length=10)
    weight_change = models.CharField(max_length=10)
    device_usage = models.CharField(max_length=10)
    caffeine_alcohol = models.CharField(max_length=10)
    concentration = models.CharField(max_length=10)

    def __str__(self):
        return self.wake_freshness,self.id
