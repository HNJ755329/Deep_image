from django.shortcuts import render
from django.shortcuts import redirect
from django.http import HttpResponse
from django.views.generic import TemplateView
from app.settings import BASE_DIR
import json

from .forms import PhotoForm
from .forms import tfkerasModelForm
from .main import MyModel
from .models import Photo
from .models import CnnModelTable
from django.views.decorators.csrf import csrf_exempt

# log
from logging import getLogger, StreamHandler, DEBUG, ERROR
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

# Create your views here.

class deep_index(TemplateView):
    def __init__(self):
        self.params={
                    'model_form': '', 
                    'image_form': '',
                    'graph': '',
                    'url': '',
                    'contents': '',
                    }
    
    def get(self, request):
        self.params['cnn_models']  = CnnModelTable.objects.all()
        self.params['model_form'] = tfkerasModelForm()
        self.params['image_form'] = PhotoForm()
        self.params['images_saved'] = Photo.objects.all().order_by('date')[::-1][:5]
        return render(request, 'deep_index.html', self.params)

class deep_model_submit(TemplateView):
    def __init__(self):
        self.params={
                    'model_form': '', 
                    'image_form': '',
                    'graph': '',
                    'url': '',
                    'contents': '',
                    'ref': '',
                    }

    def post(self, request):
        logger.debug('deep_model_submit:request.POST')
        logger.debug(request.POST)
        model_name = request.POST['tfkerasModel']
        self.model = MyModel(model_name)
        self.params['model_name'] = model_name
        self.params['graph'] = self.model.get_graph()
        self.params['contents'] = self.model.show_layers()
        self.params['ref'] = self.model.ref
        response = json.dumps(self.params)
        return HttpResponse(response)

class deep_image_submit(TemplateView):
    def __init__(self):
        self.model = MyModel()
        self.params={
                    'model_form': '', 
                    'image_form': '',
                    'graph': '',
                    'url': '',
                    'contents': '',
                    }

    def post(self, request):
        logger.debug('deep_image_submit:request.POST')
        logger.debug(request.POST)
        logger.debug('deep_image_submit:request.FILES')
        logger.debug(request.FILES)
        request.POST._mutable = True
        request.POST.pop('model')
        request.POST._mutable = False
        logger.debug('deep_image_submit:request.POST after pop model')
        logger.debug(request.POST)

        form = PhotoForm(request.POST, request.FILES)
        logger.debug(form)

        if not form.is_valid():
            raise ValueError('invalid form')
        photo = form.save()
        logger.debug(photo)
        self.params['url'] = photo.image.url
        response = json.dumps(self.params)
        logger.debug(self.params)
        return HttpResponse(response)

class deep_layer(TemplateView):
    def __init__(self):
        self.params={
                    'model_form': '', 
                    'image_form': '',
                    'graph': '',
                    'url': '',
                    'contents': '',
                    }

    def post(self, request):
        logger.debug('deep_image_submit:request.POST')
        logger.debug(request.POST)

        model_name = request.POST['model']
        request.POST._mutable = True
        request.POST.pop('model')
        request.POST._mutable = False

        logger.debug('MODEL NAME : ' + model_name)

        self.model = MyModel(model_name)

        layername = request.POST['layerid'] 
        self.params['url'] = request.POST['url'] 
        # preprocess input image data
        x = self.model.path_to_image_array(BASE_DIR + self.params['url'])
        logger.debug('Layers params : ')
        logger.debug(self.params)
        logger.debug('Layers name : ' + layername)
        logger.debug(self.model.model.name)
        logger.debug('prediction layer name : ' + str(self.model.model.layers[-1].name))
        output_shape = self.model.model.get_layer(layername).output_shape
        logger.debug('output_shape : ')
        logger.debug(output_shape)
        if layername == str(self.model.model.layers[-1].name) :
            "prediction layer only"
            output = self.model.predict(x)
            logger.debug('predict 1000 categories')
        elif len(output_shape) == 2:
            "flatten layer only"
            output = self.model.getActivations_flatten(layername, x)
            output = output.tolist()
            logger.debug(layername + ' flatten data show')
        else:
            "input layer and hidden layers"
            output = self.model.getActivations(layername, x)
            logger.debug(layername + ' image show')

        response = json.dumps({'layername': layername, 'output':output})
        return HttpResponse(response)
