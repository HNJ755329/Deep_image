from django.db import models

# Create your models here.
class Photo(models.Model):
    image = models.ImageField(upload_to='upload_image')
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.image.url) + ' uploaded : ' + str(self.date)
    
    class Meta:
        verbose_name_plural = "Uploaded images"

class LayerPhoto(models.Model):
    layer_image = models.ImageField(upload_to='layer_image')

    def __str__(self):
        return self.image.url
    
    class Meta:
        verbose_name_plural = "Layer images"

class CnnModelTable(models.Model):
    name = models.CharField(max_length=30)
    Top1Accuracy = models.CharField(max_length=10)
    Top5Accuracy = models.CharField(max_length=10)
    Parameters = models.CharField(max_length=20)
    Depth = models.CharField(max_length=5)
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name_plural = "CNN MODELS"

class tfkerasModel(models.Model):
    MODEL_CHOICES = (
        ('densenet', 'tf.keras.applications.densenet'),
        ('inception_resnet_v2', 'tf.keras.applications.inception_resnet_v2'),
        ('inception_v3', 'tf.keras.applications.inception_v3'),
        ('mobilenet', 'tf.keras.applications.mobilenet'),
        ('mobilenet_v2', 'tf.keras.applications.mobilenet_v2'),
        ('nasnet', 'tf.keras.applications.nasnet'),
        ('resnet50', 'tf.keras.applications.resnet50'),
        ('vgg16', 'tf.keras.applications.vgg16'),
        ('vgg19', 'tf.keras.applications.vgg19'),
        ('xception', 'tf.keras.applications.xception'),
    )

    tfkerasModel = models.CharField(max_length=1000, choices=MODEL_CHOICES)
    
    class Meta:
        verbose_name_plural = "tfkerasModel"