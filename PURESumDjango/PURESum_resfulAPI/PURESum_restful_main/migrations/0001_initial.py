# Generated by Django 3.0 on 2020-03-12 22:08

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MovieList',
            fields=[
                ('idx', models.AutoField(db_column='idx', primary_key=True, serialize=False)),
                ('title', models.CharField(db_column='title', max_length=200)),
                ('img', models.CharField(blank=True, db_column='img', max_length=5000, null=True)),
                ('rate', models.FloatField(blank=True, db_column='rate', null=True)),
                ('review_count', models.IntegerField(blank=True, db_column='review_count', null=True)),
            ],
        ),
    ]
