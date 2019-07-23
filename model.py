from tensorflow.python.keras.applications import inception_v3, resnet50, vgg16, xception
from tensorflow.python.keras import backend as K
import numpy as np


def get_inceptionv3_model(weights='imagenet', input_tensor=None):
    return inception_v3.InceptionV3(weights=weights, include_top=False, input_tensor=input_tensor)


def get_resnet50_model(weights='imagenet', input_tensor=None):
    return resnet50.ResNet50(weights=weights, include_top=False, input_tensor=input_tensor)


def get_vgg16_model(weights='imagenet', input_tensor=None):
    return vgg16.VGG16(weights=weights, include_top=False, input_tensor=input_tensor)


def get_xception_model(weights='imagenet', input_tensor=None):
    return xception.Xception(weights=weights, include_top=False, input_tensor=input_tensor)


def continuity_loss(x, im_height, im_width):
    assert K.ndim(x) == 4
    a = K.square(x[:, :im_height - 1, :im_width - 1, :] -
                 x[:, 1:, :im_width - 1, :])
    b = K.square(x[:, :im_height - 1, :im_width - 1, :] -
                 x[:, :im_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def define_loss_and_grads(layer_dict, input_tensor, im_size, layer_contributions, continuity, dream_l2):
    loss = K.variable(0.)
    for layer_name in layer_contributions:
        # add the L2 norm of the features of a layer to the loss
        assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
        coeff = layer_contributions[layer_name]
        x = layer_dict[layer_name].output
        shape = layer_dict[layer_name].output_shape
        # we avoid border artifacts by only involving non-border pixels in the loss
        loss = loss - coeff*K.sum(K.square(x[:, 2: shape[1] - 2, 2: shape[2] - 2, :])) / np.prod(shape[1:])

    # add continuity loss
    loss = loss + continuity*continuity_loss(input_tensor, im_size[0], im_size[1]) / np.prod(im_size)
    # add image L2 norm to loss
    loss = loss + dream_l2*K.sum(K.square(input_tensor)) / np.prod(im_size)

    grads=K.gradients(loss, input_tensor)

    output=[loss]
    if isinstance(grads, (list, tuple)):
        output += grads
    else:
        output.append(grads)

    f_outputs = K.function([input_tensor], output)

    def eval_loss_and_grads(x):
        x = x.reshape((1,) + im_size)
        outs = f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    return eval_loss_and_grads


class Evaluator(object):
    def __init__(self, eval_loss_and_grads):
        self.loss_value = None
        self.grad_values = None
        self.eval_loss_and_grads = eval_loss_and_grads

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
