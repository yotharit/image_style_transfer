from __future__ import print_function, division

import scipy.misc, numpy as np, os, sys
import random
from .coral import coral_numpy # , coral_pytorch
# from color_transfer import color_transfer


### Image helpers

def get_files(img_dir):
    paths = []
    if(os.path.isdir(content_path)):
        files = os.listdir(img_dir)
        for x in files:
            if x.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(img_dir, x))
    else:
        paths.appends(img_dir)
    return paths

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def get_img(src):
    img = scipy.misc.imread(src, mode='RGB')
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))
    return img

def center_crop(img, size=256):
    height, width = img.shape[0], img.shape[1]

    if height < size or width < size:  # Upscale to size if one side is too small
        img = resize_to(img, resize=size)
        height, width = img.shape[0], img.shape[1]

    h_off = (height - size) // 2
    w_off = (width - size) // 2
    return img[h_off:h_off+size,w_off:w_off+size]

def center_crop_to(img, H_target, W_target):
    '''Center crop a rectangle of given dimensions and resize if necessary'''
    height, width = img.shape[0], img.shape[1]

    if height < H_target or width < W_target:
        H_rat, W_rat = H_target / height, W_target / width
        rat = max(H_rat, W_rat)

        img = scipy.misc.imresize(img, rat, interp='bilinear')
        height, width = img.shape[0], img.shape[1]

    h_off = (height - H_target) // 2
    w_off = (width - W_target) // 2
    return img[h_off:h_off+H_target,w_off:w_off+W_target]

def resize_to(img, resize=512):
    '''Resize short side to target size and preserve aspect ratio'''
    height, width = img.shape[0], img.shape[1]
    if height < width:
        ratio = height / resize
        long_side = round(width / ratio)
        
    else:
        ratio = width / resize
        long_side = round(height / ratio)
    
    if(long_side<resize):
        resize_shape = (resize, long_side, 3)
        return scipy.misc.imresize(img, resize_shape, interp='bilinear')
    return img

def get_img_crop(src, resize=512, crop=256):
    '''Get & resize image and center crop'''
    img = get_img(src)
    img = resize_to(img, resize)
    return center_crop(img, crop)

def get_img_random_crop(src, resize=512, crop=256):
    '''Get & resize image and random crop'''
    img = get_img(src)
    img = resize_to(img, resize=resize)
    
    offset_h = random.randint(0, (img.shape[0]-crop))
    offset_w = random.randint(0, (img.shape[1]-crop))
    
    img = img[offset_h:offset_h+crop, offset_w:offset_w+crop, :]

    return img

def preserve_colors_np(style_rgb, content_rgb):
    coraled = coral_numpy(style_rgb/255., content_rgb/255.)
    coraled = np.uint8(np.clip(coraled, 0, 1) * 255.)
    return coraled


def swap_filter_fit(H, W, patch_size, stride, n_pools=4):
    '''Style swap may not output same size encoding if filter size > 1, calculate a new size to avoid this'''
    # Calculate size of encodings after max pooling n_pools times
    pool_out_size = lambda x: (x + 2 - 1) // 2    
    H_pool_out, W_pool_out = H, W
    for _ in range(n_pools):
        H_pool_out, W_pool_out = pool_out_size(H_pool_out), pool_out_size(W_pool_out)
    
    # Size of encoding after applying conv to determine nearest neighbor patches
    H_conv_out = (H_pool_out - patch_size) // stride + 1
    W_conv_out = (W_pool_out - patch_size) // stride + 1

    # Size after transposed conv
    H_deconv_out = (H_conv_out - 1) * stride + patch_size
    W_deconv_out = (W_conv_out - 1) * stride + patch_size

    # Stylized output size after decoding
    H_out = H_deconv_out * 2**n_pools
    W_out = W_deconv_out * 2**n_pools

    # Image will need to be resized/cropped if pooled encoding does not match style-swap encoding in either dim
    should_refit = (H_pool_out != H_deconv_out) or (W_pool_out != W_deconv_out)

    return should_refit, H_out, W_out
