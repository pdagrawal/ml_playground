from django.urls import path

from . import views

app_name = "boards"
urlpatterns = [
    path("", views.index, name="index"),
    path("new", views.new, name="new"),
    path("<id>", views.show, name="show"),
    path("<id>/edit", views.edit, name="edit"),
    path("<id>/rename", views.rename, name="rename"),
    path("<id>/share/", views.share, name="share"),
    path("<board_user_id>/remove_board_user/", views.remove_board_user, name="remove_board_user"),
    path("<id>/versions/", views.versions, name="versions"),
    path("restore_version/<version_id>", views.restore_version, name="restore_version"),
]
