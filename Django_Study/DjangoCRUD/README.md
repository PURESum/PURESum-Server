# 3. Django CRUD



# 10강. CRUD란?

## CRUD
대부분의 소프트웨어가 가지는 기본 데이터 처리 기능
`Create` 생성하기, `Read` 읽기, `Update` 수정하기, `Delete` 삭제하기

데이터베이스의 객체들을 대상으로 이루어지는 행위

Python으로 SQL을 사용하는 데이터베이스와 소통 해야 함

## ORM
Object Relational Mapping

### Create 생성하기
```python
모델명.objects.create(title="제목", content="내용")
```

### Read 읽기
```python
모델명.objects.all()
모델명.objects.first()
모델명.objects.get(id=3)
```

### Update 수정하기
```python
변수명 = 모델명.objects.get(id=3)
변수명.title = "제목 바꾸기"
변수명.save()
```

### Delete 삭제하기
```python
변수명 = 모델명.objects.get(id=3)
변수명.delete()
```



# 11강. Create 생성하기 1

게시글 업로드할 수 있는 블로그 만들기!

```python
# MyFirstDjang > urls.py

urlpatterns = [
    ...
    path('posts/', include('posts.urls')),
]


# posts > urls.py

from django.urls import path
from .views import new, create

app_name = "posts"
urlpatterns = [
    path('new/', new, name="new"),
    path('create/', create, name="create"),
]
```

```python
# posts > views.py

from django.shortcuts import render, redirect
from .models import Post

def new(request):
    return render(request, 'posts/new.html')


def create(request):
    if request.method == "POST":
        title = request.POST.get('title')
        content = request.POST.get('content')
        Post.objects.create(title=title, content=content)
        return redirect('main')   
```

```python
<!-- posts > templates > posts > new.html -->
{% extends 'base.html' %}

{% block content %}
<div class="container">
    <h1>
        새로운 글 작성하기
    </h1>
    <form action="{% url 'posts:create' %}" method="POST">
      {% csrf_token %}
      <div class="form-group">
        <label>글 제목</label>
        <input type="text" class="form-control" name="title">
      </div>
      <div class="form-group">
        <label>글 내용</label>
        <textarea class="form-control" name="content"></textarea>
      </div>
      <input type="submit" value="글 작성" class="btn btn-outline-primary">
    </form>   
</div>
{% endblock %}
```



















