import numpy as np
import PIL
import tensorflow as tf
import base64
import io 
# from IPython.display import HTML, Image, clear_output, display
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications import nasnet
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications import xception
# from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
# from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.utils import plot_model

from app.settings import BASE_DIR

from logging import getLogger, StreamHandler, DEBUG, ERROR
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(ERROR)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

class MyModel(object):
    def __init__(self, model_name='resnet50'):
        tf.compat.v1.reset_default_graph()
        session = tf.compat.v1.Session()
        self.set_model(model_name)
        # self.model.compile(optimizer='adam', loss='categorical_crossentropy' ,metrics=['accuracy'])
    
    def set_model(self, model_name, top_n=5):
        if model_name == 'densenet':
            self.model = densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
            self.target_size = (224,224)
            self.decoder = lambda x: densenet.decode_predictions(x, top=top_n)
            self.ref = """
                <ul>
                <li><a href='https://arxiv.org/abs/1608.06993' target='_blank'>
                Densely Connected Convolutional Networks</a> (CVPR 2017 Best Paper Award)</li>
                </ul>
                """

        elif model_name == 'inception_resnet_v2':
            self.model = inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
            self.target_size = (299,299)
            self.decoder = lambda x: inception_resnet_v2.decode_predictions(x, top=top_n)
            self.ref = """
                <ul>
                <li><a href='https://arxiv.org/abs/1602.07261' target='_blank'>
                Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning</a></li>
                </ul>
                """

        elif model_name == 'inception_v3':
            self.model = inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
            self.target_size = (299,299)
            self.decoder = lambda x: inception_v3.decode_predictions(x, top=top_n)
            self.ref = """<ul>
                <li><a href='https://arxiv.org/abs/1512.00567' target='_blank'>
                Rethinking the Inception Architecture for Computer Vision</a></li>
                </ul>
                """

        elif model_name == 'mobilenet':
            self.model = mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
            self.target_size = (224,224)
            self.decoder = lambda x: mobilenet.decode_predictions(x, top=top_n)
            self.ref = """<ul>
                <li><a href='https://arxiv.org/abs/1704.04861' target='_blank'>
                MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications</a></li>
                </ul>
                """

        elif model_name == 'mobilenet_v2':
            self.model = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
            self.target_size = (224,224)
            self.decoder = lambda x: mobilenet_v2.decode_predictions(x, top=top_n)
            self.ref = """<ul>
                <li><a href='https://arxiv.org/abs/1801.04381' target='_blank'>
                MobileNetV2: Inverted Residuals and Linear Bottlenecks</a></li>
                </ul>
                """

        elif model_name == 'nasnet':
            self.model = nasnet.NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
            self.target_size = (224,224)
            self.decoder = lambda x: nasnet.decode_predictions(x, top=top_n)
            self.ref = """<ul>
                <li><a href='https://arxiv.org/abs/1707.07012' target='_blank'>
                Learning Transferable Architectures for Scalable Image Recognition</a></li>
                </ul>
                """

        elif model_name == 'resnet50':
            self.model = resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
            self.target_size = (224,224)
            self.decoder = lambda x: resnet50.decode_predictions(x, top=top_n)
            self.ref = """<ul>
                <li>ResNet : 
                <a href='https://arxiv.org/abs/1512.03385' target='_blank'>Deep Residual Learning for Image Recognition
                </a></li>
                </ul>
                """

        elif model_name == 'vgg16':
            self.model = vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
            self.target_size = (224,224)
            self.decoder = lambda x: vgg16.decode_predictions(x, top=top_n)
            self.ref = """<ul>
            <li><a href='https://arxiv.org/abs/1409.1556' target='_blank'>
            Very Deep Convolutional Networks for Large-Scale Image Recognition</a></li>
            </ul>"""


        elif model_name == 'vgg19':
            self.model = vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
            self.target_size = (224,224)
            self.decoder = lambda x: vgg19.decode_predictions(x, top=top_n)
            self.ref = """<ul>
            <li><a href='https://arxiv.org/abs/1409.1556' target='_blank'>Very Deep Convolutional Networks for Large-Scale Image Recognition</a></li>
            </ul>"""

        elif model_name == 'xception':
            self.model = xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
            self.target_size = (299,299)
            self.decoder = lambda x: xception.decode_predictions(x, top=top_n)
            self.ref = """<ul>
            <li><a href='https://arxiv.org/abs/1610.02357' target='_blank'>Xception: Deep Learning with Depthwise Separable Convolutions</a></li>
            </ul>"""

        else:
            logger.ERROR('There has no model name !!!')

    def get_graph(self):
        return show_graph(tf.get_default_graph())
    
    def predict(self, image_array):
        r = []
        predict = self.model.predict(image_array, verbose=1)
        labels = self.decoder(predict)
        labels = labels[0]
        r.append('TOP 5 PREDICTIONS : ')
        for label in labels:
            r.append('%s (%.2f%%) ' % (label[1], label[2]*100))
        return r

    def show_layers(self):
        contents = []
        contents.append(
            f"""
            <div class='layer well-sm'>
                <div class='btn btn-info'>Model name</div>
                <div class='btn btn-default'>Layer name ( Click here! )</div>
                <div class='label label-default'>output_shape</div>
                <div class=''></div>
                <div class=''></div>
                <div class=''></div>
            </div>""")
        for layer in self.model.layers:
            contents.append(
                f"""
                <div id='{layer.name}' class='layer well'>
                    <div class='model_name btn btn-info'>{self.model.name}</div>
                    <div class='layername btn btn-default'>{layer.name}</div>
                    <div class='layer_config label label-default'>output_shape : {layer.output_shape}</div>
                    <div class='loading'></div>
                    <div class='layerimages'></div>
                    <div class='altshow'></div>
                </div>""")
        return contents

    def _show_layers(self):
        layers = layersname_to_list(self.model)
        contents = []
        for layer in layers:
            contents.append(
                f"""
                <div id='{layer}' class='layer well'>
                    <div class='model_name btn btn-info'>{self.model.name}</div>
                    <div class='layername btn btn-default'>{layer}</div>
                    <div class='loading'></div>
                    <div class='layerimages'></div>
                    <div class='altshow'></div>
                </div>""")
        return contents
    
    def getActivations(self, layername, imagearray):
        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            _model = feat_extraction_model(self.model, layername=layername)
            return plotNNFilter(_model(imagearray), self.model.name, layername)

    def getActivations_flatten(self, layername, imagearray):
        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            _model = feat_extraction_model(self.model, layername=layername)
            return plotNNFilter_flatten(_model(imagearray))


    def path_to_image_array(self, img_path):
        img = image.load_img(img_path, target_size=self.target_size)
        # PIL形式をnumpyのndarray形式に変換
        image_array = image.img_to_array(img,)
        # (height, width, 3) -> (1, height, width, 3)
        image_array = image_array.reshape((1,) + image_array.shape)
        # rescale
        image_array = image_array / 255.0
        return image_array

def feat_extraction_model(model, layername)->'From model.input To layername:str Model':
    output = model.get_layer(layername).output
    return tf.keras.Model(inputs=model.input, outputs=output)

def display_HTML_images(urls):
    # img_style = " margin: 1px; float: left; border: 1px solid black;"
    # images_list = ''.join([f"<img style='{img_style}' src='{u}' class='layerimage' />" for u in urls])
    images_list = ''.join([f"<img class='layerimage' src='{u}'/>" for u in urls])
    return images_list 

def display_HTML_images_base64(urls, alts):
    # img_style = " margin: 1px; float: left; border: 1px solid black;"
    # images_list = ''.join([f"<img style='{img_style}' src='data:image/jpg;base64,{u}' class='layerimage' alt='{alt}'/>" for u, alt in zip(urls, alts)])
    images_list = ''.join([f"<img class='layerimage' src='data:image/jpg;base64,{u}' alt='{alt}'/>" for u, alt in zip(urls, alts)])
    return images_list 

def plotNNFilter_flatten(units):
    flatten_array = units[0].eval()
    return flatten_array

def plotNNFilter(units, modelname, layername):
    filters = int(units.shape[3])
    image_urls = []
    alts = []
    for i in range(filters):
        img_array = units[0,:,:,i].eval()
        # img = PIL.Image.fromarray(img_array, mode='L')
        img_array = (img_array * 255.0).astype('uint8')
        # if np.mean(img_array) < 10:
        #     continue
        img = PIL.Image.fromarray(img_array)
        # img.resize((64, 64))
        buffer = io.BytesIO() # メモリ上への仮保管先を生成
        img.save(buffer, format="JPEG") # pillowのImage.saveメソッドで仮保管先へ保存
        # 保存したデータをbase64encodeメソッド読み込み
        # -> byte型からstr型に変換
        # -> 余分な区切り文字(　'　)を削除
        base64Img = base64.b64encode(buffer.getvalue()).decode().replace("'", "")
        # 保存する場合。なぜかうまくいかなかったので、修正の余地あり？
        # img.save(BASE_DIR + f'/media/layers-image/{modelname}_{layername}_{i}.jpg', 'JPEG')
        # image_urls.append(f'/media/layers-image/{modelname}_{layername}_{i}.jpg')
        image_urls.append(base64Img)
        alts.append(img_array)
        if i > 11:
            break
    return display_HTML_images_base64(image_urls, alts)

def layersname_to_list(model)->'layers_name_list':
    layers_list = []
    for layer in model.layers:
        layers_list.append(layer.name)
    return layers_list

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>"%size)
    return strip_def

def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
  
    iframe = """
        <iframe seamless style="width:840px;height:620px;border:0;margin:0px auto" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;')) # code.replace('"', '&quot;')
    return iframe