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

























