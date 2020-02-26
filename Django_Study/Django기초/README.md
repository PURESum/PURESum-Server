# 02 Django 기초



# 1강. Django란?



# 2강. MTV 패턴에 대한 이해

- Client: 클라이언트, 사용자 (고객)
"여기 짜장면 하나요!"

## Model, Template, View
- View: 요청에 대한 응답을 하는 곳. 요청을 처리해주는 곳. like 주방
"네 여기요!"

ex) 11번가 url로 접속
url 접속: 11번가 서버(View)로 `request` 요청
11번가 홈페이지가 화면에 표시됨(Template) -> 11번가 서버의 `response` 결과

- Template: View에서 response로 쓰이는 HTML 등
`render`: template을 response로 client에게 보여주는 함수

- Model: 추상적 개념. like 붕어빵 틀 (붕어빵은 Object)
DataBase에 테이블 형태로 만들기 위한 설계
column, field, attribute: 특정 모델의 속성들

#### model -> migrate -> database table

- DataBase: 실제로 데이터(객체)를 저장하는 곳
SQL이라는 언어를 씀

- ORM (Object Relational Mapping)
python과 SQL은 다른 언어. 통역사 역할

### modeling
+CreatedAt(생성), UpdatedAt(수정) 정보는 필수
+데이터 타입(자료형) 명시
+Relation: `외래키`


# 3강. Django로 모델링하기
- `__init__.py`
python 2.x 버전에서 패키지임을 알려주기 위한 모듈

- `settings.py`
Django에 대한 다양한 설정을 담당하는 부분

- `urls.py`
어플리케이션의 경로 설정을 할 수 있는 파일

- `wsgi.py`
web server gateway interface
Django 어플리케이션을 배포할 때  Nginx와 같은 웹서버와 연결해주는 파이프라인 역할

- `manage.py`
다양한 명령어를 Django에서 실행할 수 있도록 도와주는 역할

## 실행 
```
// new run django
${python.set.compiler} ${python.set.main.path} migrate && ${python.set.compiler} ${python.set.main.path} runserver 0.0.0.0:${current.using.port}

python3 /workspace/MyFirstDjango/manage.py migrate && python3 /workspace/MyFirstDjango/manage.py runserver 0.0.0.0:80
```



## 모델링을 통한 테이블 생성
1. models.py 에서 모델링
2. python manage.py makemigrations 로 주문서를 만듬
migration: 생성 될 테이블 정보를 담은 주문서
table: 모델의 설계를 그대로 데이터베이스에 저장한 형태
3. python manage.py migrate 로 주문서 내역대로 테이블 생성



## 앱 생성
```
python manage.py startapp 앱이름
```

#### MTV를 조작할 수 있는 환경이 만들어짐

- `admin.py`
관리자 페이지 설정 관련 모듈
- `views.py`
request를 처리해서 response를 내놓는 곳

### models.py 코드 작성

```python
## python manage.py startapp posts (앱생성하기)

## posts 앱 안의 models.py

from django.db import models

class Post(models.Model):
    # CharField: 문자열 데이터 타입
    # max_length: 최대 길이
    title = models.CharField(max_length=200)
    content = models.TextField(default="")
    # IntegerField: 정수형 데이터 타입
    view_count = models.IntegerField(default=0)
    
    # DateTimeField: 시간 날짜 데이터 타입
    # auto_now_add: 생성될 때 현재시간 저장
    created_at = models.DateTimeField(auto_now_add=True)
    # auto_now: 생성, 수정될 때 현재시간 저장
    updated_at = models.DateTimeField(auto_now=True)   

## settings.py 에서 INSTALLED_APPS 에 'posts' 추가하기!
```



### settings.py

INSTALLED_APPS 리스트 안에 아래 내용 추가

``` 
'앱이름', 
```



```
python manage.py makemigrations
```



### 앱이름/migrations/0001_initial.py

- id: 기본키(primary key), 고유 식별 가능한 정보
한 테이블 안에서 동일한 id(primary key)를 갖는 객체는 없어야 함



```
python manage.py migrate
```
데이터베이스에 테이블 형태로 모델이 만들어짐



### ORM

```
python manage.py shell

# 만든 모델(클래스)를 가져와야함(import)
from 앱이름.models import 클래스이름

# 생성
클래스이름.objects.create(title="제목", content="내용")

# 조회
클래스이름.objects.all()

# id=1인 객체
변수명 = 클래스이름.objects.get(id=1)

# 속성 출력
변수명.title
변수명.content
변수명.view_count
변수명.created_at
```

# 4강. 관리자 페이지 활용하기

관리자 페이지에서 CRUD, 검색, 필터링 등의 관리자 기능 만들기

### 슈퍼유저 생성
```
python manage.py createsuperuser
Username:
Email address: 
Password:
```

### admin 페이지 수정
admin.py
```python
from django.contrib import admin
from .models import 클래스이름

admin.site.register(클래스이름)
```

- 데코레이터: 함수나 클래스를 꾸며주는 역할

```python
## posts 앱안의 admin.py

from django.contrib import admin
from .models import Post

@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    # list_display: tuple 형태로 원하는 column들을 보여줌
    list_display = (
        'id', 
        'title', 
        'view_count', 
        'created_at', 
        'updated_at'
    )
    search_fields = (
        'title',
    )
    # list_filter
    # list_display_links
```



# 5강. HTML 띄우기
view, template

1. 경로 만들기
ex) naver.com -> 도메인
url = 도메인 + 경로

2. 해당 경로에 view를 할당

```python
## posts 앱 안에 templates 폴더 생성 그 안에 posts 폴더생성 그 안에 main.html 생성

## posts > views.py
# render: response로 html을 뿌려줄 때 사용하는 함수
from django.shortcuts import render

# render라는 함수에서 html 등의 파일을 가져올 때 templates 이름의 폴더에서 가져옴
# 다른 앱(폴더)에 같은 이름의 파일이 있다면 원하지 않는 파일을 가져올 수도 있으므로 주의해야 함
def main(request):
    return render(request, 'posts/main.html')
```

```python
## 프로젝트 폴더 > urls.py
from django.comtrib import admin
from django.urls import path
from posts.views import main # 도메인에서 main 함수로 가도록 설정하기 위해

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', main),
]
```

3. 해당 view에서 요청을 처리하여 응답


















