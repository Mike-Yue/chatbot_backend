from django.urls import include, path
from rest_framework import routers
from backend.api import views

router = routers.DefaultRouter()
urlpatterns = [
    path('', include(router.urls)),
    path(r'similar-symptoms', views.SimilarSymptoms.as_view(), name="similar-symptoms"),
    path(r'diagnoses', views.Diagnosis.as_view(), name="diagnosis"),
    path(r'check-symptom', views.CheckSymptom.as_view(), name="check-symptom"),
]