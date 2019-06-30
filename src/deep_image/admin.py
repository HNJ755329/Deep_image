from django.contrib import admin
from .models import Photo
from .models import CnnModelTable
from .models import CNNModels

# Register your models here.

class PhotoAdmin(admin.ModelAdmin):
    pass

class CnnModelTableAdmin(admin.ModelAdmin):
    pass

class CNNModelsAdmin(admin.ModelAdmin):
    pass

admin.site.register(Photo, PhotoAdmin)
admin.site.register(CnnModelTable, CnnModelTableAdmin)
admin.site.register(CNNModels, CNNModelsAdmin)
