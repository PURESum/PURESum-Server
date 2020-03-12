from django.conf.urls import url, include
from . import views
from django.conf import settings
from django.conf.urls.static import static
from rest_framework import routers
app_name = 'PURESum_restful_main'

router = routers.DefaultRouter()
router.register(r'movie_board', views.puresum_restful_main)
urlpatterns = [
    url('api-auth/', include('rest_framework.urls')),
    url(r'^$', include(router.urls)),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
