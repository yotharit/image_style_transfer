import os
import numpy as np
import tensorflow as tf

from keras import backend as K

import scipy
import time
import imageio
import scipy.misc
import matplotlib.pyplot as plt

from .wct import WCT
from .utils import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

class WctAlgo:   
    
    def __init__(self):      
        
        self.keep_colors = False        
        self.passes = 1
        self.alpha = 1
        self.content_size=1080
        self.style_size=450
        self.crop_size=0
        self.checkpoints = ['../WCT/relu5_1','../WCT/relu4_1','../WCT/relu3_1','../WCT/relu2_1','../WCT/relu1_1']
        self.relu_targets = ['relu5_1','relu4_1','relu3_1','relu2_1','relu1_1']
        self.vgg_path = '../WCT/vgg_normalised.t7'
        self.device = '/gpu:0'   
        self.ss_patch_size = 3
        self.ss_stride = 1
        self.swap5 = False
        self.ss_alpha = 0.6
        self.adain = False
        self.wct_model = WCT(checkpoints=self.checkpoints, 
                                    relu_targets=self.relu_targets,
                                    vgg_path=self.vgg_path, 
                                    device=self.device,
                                    ss_patch_size=self.ss_patch_size, 
                                    ss_stride=self.ss_stride)    
        
    def run(self,content_path,style_path,blended,adain,preserve,swap):
        content_fullpath = content_path 
        style_fullpath = style_path
        self.alpha=blended/10
        content_prefix, content_ext = os.path.splitext(content_fullpath)
        content_prefix = os.path.basename(content_prefix)  # Extract filename prefix without ext
        
        self.adain = adain
        self.keep_colors = preserve
        self.swap5 = swap

        content_img = get_img(content_fullpath)
        if self.content_size > 0:
            content_img = resize_to(content_img, self.content_size)

        style_prefix, _ = os.path.splitext(style_fullpath)
        style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext

        style_img = get_img(style_fullpath)

        if self.style_size > 0:
            style_img = resize_to(style_img, self.style_size)
        if self.crop_size > 0:
            style_img = center_crop(style_img, self.crop_size)

        if self.keep_colors:
            style_img = preserve_colors_np(style_img, content_img)

        # Run the frame through the style network
        stylized_rgb = self.wct_model.predict(content_img, style_img, self.alpha, self.swap5, self.ss_alpha, self.adain)

        if self.passes > 1:
            for _ in range(self.passes-1):
                stylized_rgb = self.wct_model.predict(stylized_rgb, style_img, self.alpha, self.swap5, self.ss_alpha, self.adain)   
        return stylized_rgb
