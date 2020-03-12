from django.shortcuts import render, redirect
from .models import Movie

def main(request):
    return render(request, 'practice/main.html')

def new(request):
    return render(request, 'practice/new.html')


def create(request):
    if request.method == "POST":
        title = request.POST.get('title')
        content = request.POST.get('content')
        Post.objects.create(title=title, content=content)
        return redirect('main')
