from django.contrib import admin
from .models import MovieList

# Register your models here.
@admin.register(MovieList)
class PostAdmin(admin.ModelAdmin):
    # list_display: tuple 형태로 원하는 column들을 보여줌
    list_display = (
        'idx',
        'title',
        'img',
        'rate',
        'review_count'
    )
    search_fields = (
        'title',
    )
