from django.contrib import admin
from tinymce.widgets import TinyMCE
from django.db import models

from .models import Board, BoardVersion, BoardUser

class BoardVersionAdmin(admin.ModelAdmin):
    formfield_overrides = {
        models.TextField: {'widget': TinyMCE()}
    }

admin.site.register(Board)
admin.site.register(BoardVersion, BoardVersionAdmin)
admin.site.register(BoardUser)
