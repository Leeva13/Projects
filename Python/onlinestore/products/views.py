from django.shortcuts import render

from django.views.generic import ListView
from .models import Product

from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy

# View to list all products
class ProductListView(ListView):
    model = Product
    template_name = "products/product_list.html"        
    context_object_name = "products"
    ordering = ["-created_at"]

# Check if the user is a seller
class SellerRequiredMixin(UserPassesTestMixin):
    def test_func(self):
        return self.request.user.is_authenticated and self.request.user.is_seller

# View to create a new product
class ProductCreateView(LoginRequiredMixin, SellerRequiredMixin, CreateView):
    model = Product
    fields = ["name", "description", "price", "image"]
    template_name = "products/product_form.html"

    def form_valid(self, form):
        form.instance.seller = self.request.user
        return super().form_valid(form)

# View to update an existing product
class ProductUpdateView(LoginRequiredMixin, SellerRequiredMixin, UpdateView):
    model = Product
    fields = ["name", "description", "price", "image"]
    template_name = "products/product_form.html"

    def get_queryset(self):
        # Only the seller who created the product can edit it
        return self.model.objects.filter(seller=self.request.user)

# View to delete a product
class ProductDeleteView(LoginRequiredMixin, SellerRequiredMixin, DeleteView):
    model = Product
    template_name = "products/product_confirm_delete.html"
    success_url = reverse_lazy("home")

    def get_queryset(self):
        # Only the seller who created the product can delete it
        return self.model.objects.filter(seller=self.request.user)