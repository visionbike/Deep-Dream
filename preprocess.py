import imageio
import numpy as np
import scipy.ndimage
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications import inception_v3, resnet50, vgg16, xception


def preprocess_image(im_path, im_size, model_name):
    im = image.load_img(im_path, target_size=(im_size[0], im_size[1]))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    if model_name == 'inception_v3':
        im = inception_v3.preprocess_input(im)
    elif model_name == 'resnet50':
        im = resnet50.preprocess_input(im)
    elif model_name == 'vgg16':
        im = vgg16.preprocess_input(im)
    return im


def deprocess_image(x, im_size, model_name):
    x = x.reshape((im_size[0], im_size[1], 3))
    if model_name == 'resnet50' or model_name == 'vgg16':
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]
    elif model_name == 'inception_v3':
        x /= 2.
        x += 0.5
        x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def save_image(im, im_size, model_name, fn):
    pil_im = deprocess_image(np.copy(im), im_size, model_name)
    imageio.imwrite(fn, pil_im)
