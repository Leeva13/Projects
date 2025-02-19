from django.db import models

from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    is_seller = models.BooleanField(default=False, verbose_name="Status Seller")

    def __str__(self):
        return self.username
