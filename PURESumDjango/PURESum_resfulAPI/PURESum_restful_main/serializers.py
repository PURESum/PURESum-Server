from .models import MovieList
from rest_framework import serializers

class MovieSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = MovieList
        fields = ('idx', 'title', 'img', 'rate', 'review_count')