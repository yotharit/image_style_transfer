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
import cv2 as cv


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

class Webcam:   
    
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
    
    def run(self,style_path):
        cam = VideoCameraFrame(window_name='Stylized Webcam')
        style_fullpath = style_path
        
        style_prefix, _ = os.path.splitext(style_fullpath)
        style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext

        style_img = get_img(style_fullpath)

        if self.style_size > 0:
            style_img = resize_to(style_img, self.style_size)
        if self.crop_size > 0:
            style_img = center_crop(style_img, self.crop_size)

        if self.keep_colors:
            style_img = preserve_colors_np(style_img, content_img)

        cam.open_cam_usb(1)
        do_exit = False
        while not do_exit:
            retval, frame_origin = cam.get_frame()
            image = frame_origin.copy()
            stylized_frame = self.wct_model.predict(image, style_img, self.alpha, self.swap5, self.ss_alpha, self.adain)
            key = cv.waitKey(10)
            if key == 27:  # ESC key: quit program
                do_exit = True
            cam.show_in_window(stylized_frame)

        cam.close()

class VideoCameraFrame:
    def __init__(self,window_name='Webcam Image Style Transfer'):
        self.width = 720
        self.height = 480
        self.result_frame = np.zeros((480, 720, 3), np.uint8)
        self.window_name = window_name
        self.init_window()
        self.cap = None

    def init_window(self):
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.setWindowTitle(self.window_name, self.window_name)
        cv.resizeWindow(self.window_name, self.width, self.height)
        cv.moveWindow(self.window_name, 0, 0)
        cv.imshow(self.window_name,self.result_frame)

    def open_cam_usb(self, dev=0):
        # We want to set width and height here, otherwise we could just do:
        self.cap = cv.VideoCapture(dev)
        return
        # gst_str = ('v4l2src device=/dev/video{} ! '
        #            'video/x-raw, width=(int){}, height=(int){}, '
        #            'format=(string)RGB ! '
        #            'videoconvert ! appsink').format(dev, self.width, self.height)
        # self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    def get_frame(self):

        retval, frame = self.cap.read()  # grab the next image frame from camera

        return exit, frame

    def show_in_window(self,image):
        cv.imshow(self.window_name, image)
        

    def close(self):
        self.cap.release()
        self.cap = None
        cv.destroyAllWindows()
    
    
