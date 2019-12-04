from django.urls import path
from .views import ProcessedImageView

urlpatterns = [
    path('', ProcessedImageView.as_view())
]