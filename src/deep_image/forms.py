from django import forms
from .models import Photo
from .models import tfkerasModel

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

class tfkerasModelForm(forms.ModelForm):
    class Meta:
        model = tfkerasModel
        fields = ('tfkerasModel',)
        widgets = {
            'tfkerasModel': forms.Select(attrs={
                'class': "btn btn-default",
            })
        }
        labels = {
            'tfkerasModel': 'Choose Model',
        }
        label_attrs = {
            'class': 'form-control btn btn-default'
        }