from django.urls import path

from . import views

app_name = "public"
urlpatterns = [
    path("", views.index, name="index"),
    path("about", views.about, name="about"),
    path("upload_csv", views.upload_csv, name="upload_csv"),
    path("work_in_progress", views.work_in_progress, name="work_in_progress"),
]
