from django.db import models

class Movie(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField(default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
