from django.urls import path
from . import views

urlpatterns = [
    path('detect/', views.gesture_detection_view, name='gesture_detection'),
]