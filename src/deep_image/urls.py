from django.urls import path
from .views import deep_index
from .views import deep_model_submit
from .views import deep_image_submit
from .views import deep_layer

urlpatterns = [
    path("", deep_index.as_view(), name="deep_index"),
    path("image/submit", deep_image_submit.as_view(), name="deep_image_submit"),
    path("model/submit", deep_model_submit.as_view(), name="deep_model_submit"),
    path("layer", deep_layer.as_view(), name="deep_layer"),
]