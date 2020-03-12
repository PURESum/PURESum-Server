from django.db import models

# Create your models here.
class MovieList(models.Model):
    idx = models.AutoField(db_column='idx', primary_key=True)
    title = models.CharField(db_column='title', max_length=200)
    img = models.CharField(db_column='img', max_length=5000, blank=True, null=True)
    rate = models.FloatField(db_column='rate', blank=True, null=True)
    review_count = models.IntegerField(db_column='review_count', blank=True, null=True)
