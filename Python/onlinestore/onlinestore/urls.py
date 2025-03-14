"""
URL configuration for onlinestore project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from products.views import ProductListView
from accounts.views import register

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("accounts.urls")),  # create accounts/urls.py
    path("products/", include("products.urls")),  # create products/urls.py
    path("cart/", include("cart.urls")),          # create cart/urls.py
    path("", ProductListView.as_view(), name="home"),  # main page with goods
    path("register/", register, name="register"),  # registration page
    path('', ProductListView.as_view(), name='product_list'),
]
