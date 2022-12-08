from django.urls import path

from . import views

app_name = "public"
urlpatterns = [
    path("", views.index, name="index"),
    path("about", views.about, name="about"),
    path("upload_dataset", views.upload_dataset, name="upload_dataset"),
    path("train_model", views.train_model, name="train_model"),
    path("test_model", views.test_model, name="test_model"),
    path("prediction_result", views.prediction_result, name="prediction_result"),
]
