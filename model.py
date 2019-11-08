import tensorflow as tf

from atrous_resnet50 import AtrousResNet50
from ml_atrous_vgg16 import MultievelAtrousVGG16
from atrous_xception import AtrousXception
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer
from tensorflow.keras import backend

import config

class MyModel(Model):
    def __init__(self, encoder, ds_name, phase, **kwargs):
        super().__init__(name='%s_aspp' % encoder, **kwargs)
        size = config.SPECS[ds_name]["input_size"]

        if backend.image_data_format() == 'channels_last':
            self._ch_axis = 3
            self._img_axis = (1, 2)
            input_shape = (size[0], size[1], 3)
        else:
            self._ch_axis = 1
            self._img_axis = (2, 3)
            input_shape = (3, size[0], size[1])

        self.input_layer = InputLayer(input_shape=input_shape)

        encoder_weights = "imagenet" if (ds_name == "salicon" and phase == "train") else None

        self._encoder_name = encoder
        if (encoder == "atrous_resnet"):
            self.encoder = AtrousResNet50(input_shape=self.input_layer.output_shape[0][1:], weights= encoder_weights)
        elif (encoder == "atrous_xception"):
            self.encoder = AtrousXception(input_shape=self.input_layer.output_shape[0][1:], weights= encoder_weights)
        elif (encoder == "ml_atrous_vgg"):
            self.encoder = MultievelAtrousVGG16(input_shape=self.input_layer.output_shape[0][1:], weights= encoder_weights)
        else:
            raise ValueError("encoder %s has not been implemented yet")
        
        self.aspp = self._aspp(input_shape=self.encoder.output_shape[1:])
        self.decoder = self._decoder(input_shape=self.aspp.output_shape[1:])

        self.build(self.input_layer.output_shape[0])

    def call(self, x):
        x = self.input_layer(x)
        x = self.encoder(x)
        x = self.aspp(x)
        x = self.decoder(x)
        return x
    
    def _aspp(self, input_shape=None):
        """Initialize the ASPP module samples information at multiple spatial scales in
           parallel via convolutional layers with different dilation factors.
           The activations are then combined with global scene context and
           represented as a common tensor.

        Args:
            input_tensor: input tensor
        """

        input_tensor = Input(shape=input_shape)

        # (strides, dilation_rate)
        params = [ (1, 1), (3, 4), (3, 8), (3, 12) ]

        branches = []

        # branch 1 to 4
        for i, (strides, dilation_rate) in enumerate(params):
            branches.append(Conv2D(256, strides,
                                    padding="same",
                                    dilation_rate=dilation_rate, 
                                    activation=tf.nn.relu,
                                    name="conv1_" + str(i + 1))(input_tensor))

        # branch 5
        branch5 = tf.reduce_mean(input_tensor, axis=self._img_axis, keepdims=True)
        branch5 = Conv2D(256, 1, padding="valid", activation=tf.nn.relu, name="conv1_5")(branch5)

        input_shape = input_tensor.shape.as_list()
        h_axis = self._img_axis[0]
        w_axis = self._img_axis[1]
        
        if backend.image_data_format() == 'channels_first':
            branch5 = tf.transpose(branch5, (0, 2, 3, 1))

        branch5 = tf.image.resize(branch5, (input_shape[h_axis], input_shape[w_axis]), method=tf.image.ResizeMethod.BILINEAR)

        if backend.image_data_format() == 'channels_first':
            branch5 = tf.transpose(branch5, (0, 3, 1, 2))

        branches.append(branch5)
        
        x = Conv2D(256, 1,
                        padding="same",
                        activation=tf.nn.relu,
                        name="conv2")(tf.concat(branches, axis=self._ch_axis))
        return Model(input_tensor, x, name="aspp")

    def _decoder(self, input_shape=None):
        """Initialize the decoder model applies a series of 3 upsampling blocks that each
           performs bilinear upsampling followed by a 3x3 convolution to avoid
           checkerboard artifacts in the image space. Unlike all other layers,
           the output of the model is not modified by a ReLU.

        Args:
            input_tensor (tensor, float32): input tensor.
        """
        
        input_tensor = Input(shape=input_shape)
        x = input_tensor

        kernel_sizes = [128, 64, 32]

        for i, kernel_size in enumerate(kernel_sizes):            
            x = UpSampling2D(2, interpolation="bilinear")(x)

            x = Conv2D(kernel_size, 3, padding="same", activation=tf.nn.relu,
                        name="conv" + str(i + 1))(x)

        x = Conv2D(1, 3, padding="same", name="conv4")(x)
        x = self._normalize(x)

        return Model(input_tensor, x, name="decoder")

    def _normalize(self, maps, eps=1e-7):
        """Initialize this function normalizes the output values to a range
           between 0 and 1 per saliency map.

        Args:
            maps (tensor, float32): A 4D tensor that holds the model output.
            eps (scalar, float, optional): A small factor to avoid numerical
                                           instabilities. Defaults to 1e-7.
        """

        min_per_image = tf.reduce_min(maps, axis=(1, 2, 3), keepdims=True)
        maps -= min_per_image
        max_per_image = tf.reduce_max(maps, axis=(1, 2, 3), keepdims=True)
        return tf.divide(maps, eps + max_per_image, name="output")

    def freeze_unfreeze_encoder_trained_layers(self, freeze=True):
        encoder_name = self._encoder_name
        if(encoder_name == "atrous_resnet"):
            n_of_trained_layers = 81
        elif(encoder_name == "atrous_xception"):
            n_of_trained_layers = 32
        elif(encoder_name == "ml_atrous_vgg"):
            n_of_trained_layers = 14
        
        for layer in self.encoder.layers[:n_of_trained_layers]:
            layer.trainable = (not freeze)