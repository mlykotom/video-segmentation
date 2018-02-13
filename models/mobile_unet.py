import keras.backend as K
from keras import Input
from keras.applications import mobilenet
from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import Conv2D, BatchNormalization, Activation, concatenate, Conv2DTranspose, Reshape
from keras.models import Model

from models.base_model import BaseModel
from models.layers import BilinearUpSampling2D


class MobileUNet(BaseModel):

    def __init__(self, target_size, n_classes, alpha=1.0, alpha_up=1.0, depth_multiplier=1, dropout=1e-3,
                 is_debug=False):
        self.alpha = alpha
        self.alpha_up = alpha_up
        self.depth_multiplier = depth_multiplier
        self.dropout = dropout

        super(MobileUNet, self).__init__(target_size, n_classes, is_debug)

    @staticmethod
    def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), block_id=1):
        """Adds an initial convolution layer (with batch normalization and relu6).

        # Arguments
            inputs: Input tensor of shape `(rows, cols, 3)`
                (with `channels_last` data format) or
                (3, rows, cols) (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 32.
                E.g. `(224, 224, 3)` would be one valid value.
            filters: Integer, the dimensionality of the output space
                (i.e. the number output of filters in the convolution).
            alpha: controls the width of the network.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                     are used at each layer.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.

        # Input shape
            4D tensor with shape:
            `(samples, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(samples, rows, cols, channels)` if data_format='channels_last'.

        # Output shape
            4D tensor with shape:
            `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.

        # Returns
            Output tensor of block.
        """
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        filters = int(filters * alpha)
        x = Conv2D(filters, kernel,
                   padding='same',
                   use_bias=False,
                   strides=strides,
                   name='conv_%d' % block_id)(inputs)
        x = BatchNormalization(axis=channel_axis, name='conv_%d_bn' % block_id)(x)
        return Activation(mobilenet.relu6, name='conv_%d_relu' % block_id)(x)

    @staticmethod
    def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
        """Adds a depthwise convolution block.

        A depthwise convolution block consists of a depthwise conv,
        batch normalization, relu6, pointwise convolution,
        batch normalization and relu6 activation.

        # Arguments
            inputs: Input tensor of shape `(rows, cols, channels)`
                (with `channels_last` data format) or
                (channels, rows, cols) (with `channels_first` data format).
            pointwise_conv_filters: Integer, the dimensionality of the output space
                (i.e. the number output of filters in the pointwise convolution).
            alpha: controls the width of the network.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                     are used at each layer.
            depth_multiplier: The number of depthwise convolution output channels
                for each input channel.
                The total number of depthwise convolution output
                channels will be equal to `filters_in * depth_multiplier`.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
            block_id: Integer, a unique identification designating the block number.

        # Input shape
            4D tensor with shape:
            `(batch, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, rows, cols, channels)` if data_format='channels_last'.

        # Output shape
            4D tensor with shape:
            `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.

        # Returns
            Output tensor of block.
        """
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)

        x = DepthwiseConv2D((3, 3),
                            padding='same',
                            depth_multiplier=depth_multiplier,
                            strides=strides,
                            use_bias=False,
                            name='conv_dw_%d' % block_id)(inputs)
        x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
        x = Activation(mobilenet.relu6, name='conv_dw_%d_relu' % block_id)(x)

        x = Conv2D(pointwise_conv_filters, (1, 1),
                   padding='same',
                   use_bias=False,
                   strides=(1, 1),
                   name='conv_pw_%d' % block_id)(x)
        x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
        return Activation(mobilenet.relu6, name='conv_pw_%d_relu' % block_id)(x)

    def _create_model(self):
        img_input = Input(shape=(self.target_size[0], self.target_size[1], 3))

        b00 = self._conv_block(img_input, 32, self.alpha, strides=(2, 2), block_id=0)
        b01 = self._depthwise_conv_block(b00, 64, self.alpha, self.depth_multiplier, block_id=1)

        b02 = self._depthwise_conv_block(b01, 128, self.alpha, self.depth_multiplier, block_id=2, strides=(2, 2))
        b03 = self._depthwise_conv_block(b02, 128, self.alpha, self.depth_multiplier, block_id=3)

        b04 = self._depthwise_conv_block(b03, 256, self.alpha, self.depth_multiplier, block_id=4, strides=(2, 2))
        b05 = self._depthwise_conv_block(b04, 256, self.alpha, self.depth_multiplier, block_id=5)

        b06 = self._depthwise_conv_block(b05, 512, self.alpha, self.depth_multiplier, block_id=6, strides=(2, 2))
        b07 = self._depthwise_conv_block(b06, 512, self.alpha, self.depth_multiplier, block_id=7)
        b08 = self._depthwise_conv_block(b07, 512, self.alpha, self.depth_multiplier, block_id=8)
        b09 = self._depthwise_conv_block(b08, 512, self.alpha, self.depth_multiplier, block_id=9)
        b10 = self._depthwise_conv_block(b09, 512, self.alpha, self.depth_multiplier, block_id=10)
        b11 = self._depthwise_conv_block(b10, 512, self.alpha, self.depth_multiplier, block_id=11)

        b12 = self._depthwise_conv_block(b11, 1024, self.alpha, self.depth_multiplier, block_id=12, strides=(2, 2))
        b13 = self._depthwise_conv_block(b12, 1024, self.alpha, self.depth_multiplier, block_id=13)
        # if not self.is_debug:
        #     b13 = Dropout(self.dropout)(b13)

        filters = int(512 * self.alpha)
        up1 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b13),
            b11,
        ], axis=3)
        b14 = self._depthwise_conv_block(up1, filters, self.alpha_up, self.depth_multiplier, block_id=14)

        filters = int(256 * self.alpha)
        up2 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b14),
            b05,
        ], axis=3)
        b15 = self._depthwise_conv_block(up2, filters, self.alpha_up, self.depth_multiplier, block_id=15)

        filters = int(128 * self.alpha)
        up3 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b15),
            b03,
        ], axis=3)
        b16 = self._depthwise_conv_block(up3, filters, self.alpha_up, self.depth_multiplier, block_id=16)

        filters = int(64 * self.alpha)
        up4 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b16),
            b01,
        ], axis=3)
        b17 = self._depthwise_conv_block(up4, filters, self.alpha_up, self.depth_multiplier, block_id=17)

        filters = int(32 * self.alpha)
        up5 = concatenate([b17, b00], axis=3)
        # b18 = self._depthwise_conv_block(up5, filters, self.alpha_up, self.depth_multiplier, block_id=18)
        b18 = self._conv_block(up5, filters, self.alpha_up, block_id=18)

        x = Conv2D(self.n_classes, (1, 1), kernel_initializer='he_normal', activation='linear')(b18)
        x = BilinearUpSampling2D(size=(2, 2))(x)

        x = Reshape((-1, self.n_classes))(x)
        x = Activation('softmax')(x)

        return Model(img_input, x)

    custom_objects = {
        'relu6': mobilenet.relu6,
        'DepthwiseConv2D': mobilenet.DepthwiseConv2D,
        'BilinearUpSampling2D': BilinearUpSampling2D,
    }
