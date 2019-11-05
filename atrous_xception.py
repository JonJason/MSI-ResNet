"""Xception V1 model for Keras.

On ImageNet, this model gets to a top-1 validation accuracy of 0.790
and a top-5 validation accuracy of 0.945.

Do note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function
is also different (same as Inception V3).

# Reference

- [Xception: Deep Learning with Depthwise Separable Convolutions](
    https://arxiv.org/abs/1610.02357) (CVPR 2017)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, add
from tensorflow.keras.layers import Conv2D, SeparableConv2D, GlobalMaxPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.utils as keras_utils
import tensorflow.keras.models as models
from tensorflow.keras import backend

from keras_applications import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape

TF_WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.4/'
    'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')


def AtrousXception(weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             **kwargs):
    """Instantiates the Xception architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Note that the default input image size for this model is 299x299.

    # Arguments
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=71,
                                      data_format=backend.image_data_format(),
                                      require_flatten=False,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    x = Conv2D(32, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1')(img_input)
    x = BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = MaxPool2D((3, 3),
                    strides=(2, 2),
                    padding='same',
                    name='block2_pool')(x)
    x = add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = MaxPool2D((3, 3), strides=(2, 2),
                    padding='same',
                    name='block3_pool')(x)
    x = add([x, residual])

    residual = Conv2D(728, (1, 1),
                             dilation_rate=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = MaxPool2D((3, 3), strides=(1, 1),
                    padding='same',
                    name='block4_pool')(x)
    x = add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv3_bn')(x)

        x = add([x, residual])

    residual = Conv2D(1024, (1, 1), dilation_rate=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = MaxPool2D((3, 3),
                    strides=(1, 1),
                    padding='same',
                    name='block13_pool')(x)
    x = add([x, residual])

    x = SeparableConv2D(1536, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    # additional layer (additional)
    # x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = SeparableConv2D(512, 3, padding='same', activation='relu')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPool2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='xception')

    # Load weights.
    if weights == 'imagenet':
        weights_path = keras_utils.get_file(
            'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='b0042744bf5b25fce3cb969f33bebb97')
        model.load_weights(weights_path, by_name=True)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)

if __name__ == "__main__":
    x_shape = (240, 320, 3)
    base_model = AtrousXception(input_shape=x_shape, weights=None)
    # for idx, layer in enumerate(base_model.layers):
    #     print(str(idx) + '.', layer.name, str(layer.trainable))
    base_model.summary()