from django.urls import path
from .views import ProductListView, ProductCreateView, ProductUpdateView, ProductDeleteView

urlpatterns = [
    path("add/", ProductCreateView.as_view(), name="product_add"),
    path("<int:pk>/edit/", ProductUpdateView.as_view(), name="product_edit"),
    path("<int:pk>/delete/", ProductDeleteView.as_view(), name="product_delete"),
    # Можна додати детальний перегляд товару
]
