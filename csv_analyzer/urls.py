from django.urls import path
from . import views
import csv_analyzer.views
print(dir(csv_analyzer.views))  # Vérifiez si `upload_cs
print(dir(views))
urlpatterns = [
    path('', views.upload_csv, name='upload_csv'),
    path('statistics/', views.statistical_analysis, name='statistical_analysis'),
    path('visualization/', views.visualization, name='visualization'),
    path('recherche/', views.recherche, name='recherche'),
    path('data/', views.data_preview, name='data_preview'),
    path('menu/', views.menu, name='menu'),
 
]




