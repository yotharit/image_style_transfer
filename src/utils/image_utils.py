from keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image

class ImageUtils():
    
        
    def __init__(self):
        self.IMAGE_SIZE = 512
    
    def load_img(self,path_to_image):
        # Open Image
        img = Image.open(path_to_image)
        # Calculating scale for resizing
        max_dim = self.IMAGE_SIZE
        long = max(img.size)
        scale = max_dim/long
        # Resize the image
        img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
        # Convert to array
        img = keras_image.img_to_array(img)
        # Broadcast the image array such that it has a batch dimension
        img = np.expand_dims(img, axis = 0)
        return img