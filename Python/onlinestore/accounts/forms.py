from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser

class CustomUserCreationForm(UserCreationForm):
    is_seller = forms.BooleanField(required=False, label="Register as Seller")

    class Meta:
        model = CustomUser
        fields = ("username", "email", "is_seller", "password1", "password2")
