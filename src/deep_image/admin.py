from django.contrib import admin
from .models import Photo
from .models import LayerPhoto
from .models import CnnModelTable
from .models import tfkerasModel

# Register your models here.

class PhotoAdmin(admin.ModelAdmin):
    pass

class LayerPhotoAdmin(admin.ModelAdmin):
    pass

class CnnModelTableAdmin(admin.ModelAdmin):
    pass

class tfkerasModelAdmin(admin.ModelAdmin):
    pass

admin.site.register(Photo, PhotoAdmin)
admin.site.register(LayerPhoto, LayerPhotoAdmin)
admin.site.register(CnnModelTable, CnnModelTableAdmin)
admin.site.register(tfkerasModel, tfkerasModelAdmin)
