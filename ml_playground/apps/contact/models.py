from django.db import models

# Create your models here.
class Contact(models.Model):
    name = models.CharField(max_length=200)
    email = models.EmailField(max_length=100)
    message = models.TextField(max_length=1000)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name
