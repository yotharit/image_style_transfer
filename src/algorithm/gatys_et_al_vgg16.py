import numpy as np
from PIL import Image

import tensorflow as tf

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image as keras_image

import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import fmin_l_bfgs_b

class Evaluator:
    
    def __init__(self,parent):
        self.parent = parent

    def loss(self, x):
        loss, gradients = self.parent.evaluate_loss_and_gradients(x)
        self._gradients = gradients
        return loss

    def gradients(self, x):
        return self._gradients

class Gatys:   
    
    def __init__(self):
        
        self.IMAGE_SIZE = 512
        self.ITERATIONS = 10
        self.CHANNELS = 3
        self.IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]
        self.CONTENT_WEIGHT = 0.02
        self.STYLE_WEIGHT = 4.5
        self.TOTAL_VARIATION_WEIGHT = 0.995
        self.TOTAL_VARIATION_LOSS_FACTOR = 1.25
    
        self.content_layer = "block2_conv2"
        self.style_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
    
    def load_and_process_img(self,path_to_image):
        # Open Image
        img = Image.open(path_to_image)
        # Resize the image
        img = img.resize((self.IMAGE_WIDTH, self.IMAGE_HEIGHT), Image.ANTIALIAS)
        # Normalization and Reshaping to BGR
        img = keras_image.img_to_array(img, dtype = 'float32')
        img = np.expand_dims(img,axis = 0)
        img[:,:,:,0] -= self.IMAGENET_MEAN_RGB_VALUES[2]
        img[:,:,:,1] -= self.IMAGENET_MEAN_RGB_VALUES[1]
        img[:,:,:,2] -= self.IMAGENET_MEAN_RGB_VALUES[0]
        img = img[:,:,:,::-1]
        return img
    
    def content_loss(self,content, combination):
        return backend.sum(backend.square(combination - content))
    
    def gram_matrix(self,x):
        features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
        gram = backend.dot(features, backend.transpose(features))
        return gram
    
    def compute_style_loss(self,style, combination):
        style = self.gram_matrix(style)
        combination = self.gram_matrix(combination)
        size = self.IMAGE_HEIGHT * self.IMAGE_WIDTH
        return backend.sum(backend.square(style - combination)) / (4. * (self.CHANNELS ** 2) * (size ** 2))
    
    def total_variation_loss(self,x):
        a = backend.square(x[:, :self.IMAGE_HEIGHT-1, :self.IMAGE_WIDTH-1, :] - x[:, 1:, :self.IMAGE_WIDTH-1, :])
        b = backend.square(x[:, :self.IMAGE_HEIGHT-1, :self.IMAGE_WIDTH-1, :] - x[:, :self.IMAGE_HEIGHT-1, 1:, :])
        return backend.sum(backend.pow(a + b, self.TOTAL_VARIATION_LOSS_FACTOR))
    
    def evaluate_loss_and_gradients(self,x):
        x = x.reshape((1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS))
        outs = backend.function([self.combination_image], self.outputs)([x])
        loss = outs[0]
        gradients = outs[1].flatten().astype("float64")
        return loss, gradients
    
    def deprocessing_image(self,input_image):
        image = input_image.copy()
        image = image.reshape((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS))
        image = image[:, :, ::-1]
        image[:, :, 0] += self.IMAGENET_MEAN_RGB_VALUES[2]
        image[:, :, 1] += self.IMAGENET_MEAN_RGB_VALUES[1]
        image[:, :, 2] += self.IMAGENET_MEAN_RGB_VALUES[0]
        image = np.clip(image, 0, 255).astype("uint8")
        return image
    
    def run(self,content_path,style_path):
        
        backend.clear_session()
        temp = Image.open(content_path)
        # Calculating scale for resizing
        long = max(temp.size)
        scale = self.IMAGE_SIZE/long
        self.IMAGE_WIDTH = round(temp.size[0]*scale)
        self.IMAGE_HEIGHT = round(temp.size[1]*scale)
        
        input_image_array = self.load_and_process_img(content_path)
        style_image_array = self.load_and_process_img(style_path)
        
        self.input_image = backend.variable(input_image_array)
        self.style_image = backend.variable(style_image_array)
        self.combination_image = backend.placeholder((1,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,3))
        input_tensor = backend.concatenate([self.input_image,self.style_image,self.combination_image], axis=0)
        
        model = VGG16(input_tensor = input_tensor, include_top = False)
        
        layers = dict([(layer.name, layer.output) for layer in model.layers])
        layer_features = layers[self.content_layer]
        
        content_image_features = layer_features[0,:,:,:]
        combination_features = layer_features[2,:,:,:]
   
        loss = backend.variable(0.)
        loss = loss + self.CONTENT_WEIGHT * self.content_loss(content_image_features,combination_features)
        
        for layer_name in self.style_layers:
            layer_features = layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            style_loss = self.compute_style_loss(style_features, combination_features)
            loss += (self.STYLE_WEIGHT / len(self.style_layers)) * style_loss
        
        loss += self.TOTAL_VARIATION_WEIGHT * self.total_variation_loss(self.combination_image)
        evaluator = Evaluator(self)
        
        self.outputs = [loss]
        self.outputs += backend.gradients(loss, self.combination_image)
        
        x = np.random.uniform(0, 255, (1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)) - 128.
        for i in range(self.ITERATIONS):
            x, loss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients, maxfun=20)
            print("Iteration %d completed with loss %d" % (i, loss))

        output_image = self.deprocessing_image(x)
        output_image = keras_image.array_to_img(output_image)
        return output_image