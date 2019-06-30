from django import forms
from .models import Photo
from .models import CNNModels

class PhotoForm(forms.ModelForm):
    class Meta:
        model = Photo
        fields = ('image',)
        widgets = {
            'image': forms.FileInput(attrs={
                # 'class': "form-control btn btn-default",
                'style': "display: none;",
            })
        }
        labels = {
            'image': 'Choose Image',
        }
        label_attrs = {
            'class': 'form-control btn btn-default'
        }

class CNNModelsForm(forms.ModelForm):
    class Meta:
        model = CNNModels
        fields = ('CNNModels',)
        widgets = {
            'CNNModels': forms.Select(attrs={
                'class': "btn btn-default",
            })
        }
        labels = {
            'CNNModels': 'Choose Model',
        }
        label_attrs = {
            'class': 'form-control btn btn-default'
        }