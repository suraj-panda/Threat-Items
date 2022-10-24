"""threat URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from firstPage import views
from django.conf import settings

#from django.views.static import static
from django.views.static import serve

urlpatterns = [
    path('admin/', admin.site.urls),
    url('^$', views.index, name='Homepage'),
    url('custom1',views.custom1, name='custom1'),
    url('custom2',views.custom2, name='custom2'),
    url('predict1',views.predict1, name='predict1'),
    url('predict2',views.predict2, name='predict2'),
    url(r'^basic/(?P<path>.*)$', serve, {'document_root': settings.BASIC_ROOT}),
    url(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
    url(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATIC_ROOT}),
    url(r'^results/(?P<path>.*)$', serve, {'document_root': settings.RESULT_ROOT}),
    url(r'^merge/(?P<path>.*)$', serve, {'document_root': settings.MERGE_ROOT}),
]
# urlpatterns=urlpatterns+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# urlpatterns=urlpatterns+static(settings.BASIC_URL, document_root=settings.BASIC_ROOT)
# urlpatterns=urlpatterns+static(settings.RESULT_URL, document_root=settings.RESULT_ROOT)
# urlpatterns=urlpatterns+static(settings.MERGE_URL, document_root=settings.MERGE_ROOT)