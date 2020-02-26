from django.contrib import admin
from .models import Movie

@admin.register(Movie)
class PostAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'title',
        'content',
        'created_at',
        'updated_at'
    )
    search_fields = (
        'title',
    )
