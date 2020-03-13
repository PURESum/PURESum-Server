from django.shortcuts import render
from .models import MovieList
from .serializers import MovieSerializer
from django.views import View
from django.views import generic
from rest_framework import viewsets

# Create your views here.

class puresum_restful_main(viewsets.ModelViewSet):
    queryset = MovieList.objects.all()
    serializer_class = MovieSerializer
