from django.urls import path
from .views import new, create

app_name = "practice"
urlpatterns = [
    path('new/', new, name="new"),
    path('create/', create, name="create"),
]