import keras
import keras.backend as K
import tensorflow as tf
from keras import Input
from keras.applications import mobilenet
from keras.applications.mobilenet import DepthwiseConv2D
from keras.engine import Layer
from keras.layers import Conv2D, BatchNormalization, Activation, concatenate, Conv2DTranspose, Reshape, Lambda, Add
from keras.models import Model

from base_model import BaseModel
from layers import BilinearUpSampling2D
from layers import tf_warp


class FlowFilter(Layer):
    def __init__(self, init_value=1.0, **kwargs):
        self.init_value = init_value
        super(FlowFilter, self).__init__(**kwargs)

    def build(self, input_shape):
        self.fil = self.add_weight(name='filter',
                                   shape=(1,),
                                   initializer=keras.initializers.Constant(self.init_value),
                                   trainable=True)

        super(FlowFilter, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return K.tf.where(K.tf.greater(K.tf.abs(inputs), self.fil), inputs, inputs)


class MobileUNetWarp(BaseModel):
    warp_decoder = []

    def __init__(self, target_size, n_classes, alpha=1.0, alpha_up=1.0, depth_multiplier=1, dropout=1e-3,
                 is_debug=False):
        self.alpha = alpha
        self.alpha_up = alpha_up
        self.depth_multiplier = depth_multiplier
        self.dropout = dropout
        super(MobileUNetWarp, self).__init__(target_size, n_classes, is_debug)

    custom_objects = {
        'relu6': mobilenet.relu6,
        'DepthwiseConv2D': mobilenet.DepthwiseConv2D,

        'BilinearUpSampling2D': BilinearUpSampling2D,
    }

    @staticmethod
    def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), block_id=1, prefix=''):
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
                   name='%sconv_%d' % (prefix, block_id))(inputs)
        x = BatchNormalization(axis=channel_axis, name='%sconv_%d_bn' % (prefix, block_id))(x)
        return Activation(mobilenet.relu6, name='%sconv_%d_relu' % (prefix, block_id))(x)

    @staticmethod
    def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1,
                              prefix=''):
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
                            name='%sconv_dw_%d' % (prefix, block_id))(inputs)
        x = BatchNormalization(axis=channel_axis, name='%sconv_dw_%d_bn' % (prefix, block_id))(x)
        x = Activation(mobilenet.relu6, name='%sconv_dw_%d_relu' % (prefix, block_id))(x)

        x = Conv2D(pointwise_conv_filters, (1, 1),
                   padding='same',
                   use_bias=False,
                   strides=(1, 1),
                   name='%sconv_pw_%d' % (prefix, block_id))(x)
        x = BatchNormalization(axis=channel_axis, name='%sconv_pw_%d_bn' % (prefix, block_id))(x)
        return Activation(mobilenet.relu6, name='%sconv_pw_%d_relu' % (prefix, block_id))(x)

    def netwarp_module(self, img_old, img_new, flo, diff):
        x = concatenate([img_old, img_new, flo, diff])
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
        x = concatenate([flo, x])
        x = BatchNormalization()(x)
        x = Conv2D(2, (3, 3), padding='same', name='transformed_flow')(x)
        return x

    def warp(self, x):
        img = x[0]
        flow = x[1]
        flow = FlowFilter()(flow)

        out_size = img.get_shape().as_list()[1:3]
        resized_flow = Lambda(lambda image: tf.image.resize_bilinear(image, out_size))(flow)
        out = tf_warp(img, resized_flow, out_size)
        out = BatchNormalization()(out)
        return out

    def frame_branch(self, img_input, prefix=''):
        b00 = self._conv_block(img_input, 32, self.alpha, strides=(2, 2), block_id=0, prefix=prefix)
        # if not self.is_debug:
        #     b00 = SpatialDropout2D(0.2)(b00)

        b01 = self._depthwise_conv_block(b00, 64, self.alpha, self.depth_multiplier, block_id=1, prefix=prefix)
        # --
        b02 = self._depthwise_conv_block(b01, 128, self.alpha, self.depth_multiplier, block_id=2, strides=(2, 2), prefix=prefix)
        b03 = self._depthwise_conv_block(b02, 128, self.alpha, self.depth_multiplier, block_id=3, prefix=prefix)
        # --
        b04 = self._depthwise_conv_block(b03, 256, self.alpha, self.depth_multiplier, block_id=4, strides=(2, 2), prefix=prefix)
        b05 = self._depthwise_conv_block(b04, 256, self.alpha, self.depth_multiplier, block_id=5, prefix=prefix)

        # if not self.is_debug:
        #     b05 = SpatialDropout2D(0.2)(b05)

        # --
        b06 = self._depthwise_conv_block(b05, 512, self.alpha, self.depth_multiplier, block_id=6, strides=(2, 2), prefix=prefix)
        b07 = self._depthwise_conv_block(b06, 512, self.alpha, self.depth_multiplier, block_id=7, prefix=prefix)
        b08 = self._depthwise_conv_block(b07, 512, self.alpha, self.depth_multiplier, block_id=8, prefix=prefix)
        b09 = self._depthwise_conv_block(b08, 512, self.alpha, self.depth_multiplier, block_id=9, prefix=prefix)
        b10 = self._depthwise_conv_block(b09, 512, self.alpha, self.depth_multiplier, block_id=10, prefix=prefix)
        b11 = self._depthwise_conv_block(b10, 512, self.alpha, self.depth_multiplier, block_id=11, prefix=prefix)

        # if not self.is_debug:
        #     b11 = SpatialDropout2D(0.1)(b11)

        # --
        b12 = self._depthwise_conv_block(b11, 1024, self.alpha, self.depth_multiplier, block_id=12, strides=(2, 2), prefix=prefix)
        b13 = self._depthwise_conv_block(b12, 1024, self.alpha, self.depth_multiplier, block_id=13, prefix=prefix)

        # if not self.is_debug:
        #     b13 = SpatialDropout2D(0.1)(b13)
        # b13 = Dropout(0.2)

        return b00, b01, b03, b05, b11, b13

    def decoder(self):
        filters = int(512 * self.alpha)

        b11 = BatchNormalization()(Add()([self.b11, self.warped4])) if 4 in self.warp_decoder else self.b11

        up1 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(self.b13),
            b11,
        ], axis=3)

        b14 = self._depthwise_conv_block(up1, filters, self.alpha_up, self.depth_multiplier, block_id=14)

        filters = int(256 * self.alpha)

        b05 = BatchNormalization()(Add()([self.b05, self.warped3])) if 3 in self.warp_decoder else self.b05

        up2 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b14),
            b05,
        ], axis=3)

        b15 = self._depthwise_conv_block(up2, filters, self.alpha_up, self.depth_multiplier, block_id=15)

        filters = int(128 * self.alpha)

        b03 = BatchNormalization()(Add()([self.b03, self.warped2])) if 2 in self.warp_decoder else self.b03

        up3 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b15),
            b03,
        ], axis=3)

        b16 = self._depthwise_conv_block(up3, filters, self.alpha_up, self.depth_multiplier, block_id=16)

        filters = int(64 * self.alpha)

        b01 = BatchNormalization()(Add()([self.b01, self.warped1])) if 1 in self.warp_decoder else self.b01

        up4 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b16),
            b01
        ], axis=3)

        b17 = self._depthwise_conv_block(up4, filters, self.alpha_up, self.depth_multiplier, block_id=17)

        filters = int(32 * self.alpha)

        b00 = BatchNormalization()(Add()([self.b00, self.warped0])) if 0 in self.warp_decoder else self.b00

        up5 = concatenate([
            b17,
            b00
        ], axis=3)
        up5 = BatchNormalization()(up5)

        b18 = self._depthwise_conv_block(up5, filters, self.alpha_up, self.depth_multiplier, block_id=18)
        b18 = self._conv_block(b18, filters, self.alpha_up, block_id=18)

        return b18

    def _create_model(self):
        img_old = Input(shape=self.target_size + (3,), name='data_old')
        img_new = Input(shape=self.target_size + (3,), name='data_new')
        flo = Input(shape=self.target_size + (2,), name='data_flow')
        diff = Input(shape=self.target_size + (3,), name='data_diff')

        all_inputs = [img_old, img_new, flo, diff]
        transformed_flow = self.netwarp_module(img_old, img_new, flo, diff)

        # -------- OLD FRAME BRANCH
        self.old_b00, self.old_b01, self.old_b03, self.old_b05, self.old_b11, self.old_b13 = self.frame_branch(img_new, prefix='old_')

        # -------- ACTUAL FRAME BRANCH
        self.b00, self.b01, self.b03, self.b05, self.b11, self.b13 = self.frame_branch(img_new)

        # -------- WARPING
        self.warped0 = Lambda(self.warp, name="warp1")([self.old_b00, transformed_flow])
        self.warped1 = Lambda(self.warp, name="warp1")([self.old_b01, transformed_flow])
        self.warped2 = Lambda(self.warp, name="warp2")([self.old_b03, transformed_flow])
        self.warped3 = Lambda(self.warp, name="warp3")([self.old_b05, transformed_flow])
        self.warped4 = Lambda(self.warp, name="warp4")([self.old_b11, transformed_flow])
        self.warped5 = Lambda(self.warp, name="warp5")([self.old_b13, transformed_flow])

        # -------- DECODER
        x = self.decoder()

        x = Conv2D(self.n_classes, (1, 1), kernel_initializer='he_normal', activation='linear')(x)
        x = BilinearUpSampling2D(size=(2, 2))(x)

        x = Reshape((-1, self.n_classes))(x)
        x = Activation('softmax')(x)

        return Model(all_inputs, x)


class MobileUNetWarp4(MobileUNetWarp):
    def __init__(self, target_size, n_classes, is_debug=False):
        self.warp_decoder.append(4)
        super(MobileUNetWarp4, self).__init__(target_size, n_classes, is_debug=is_debug)


class MobileUNetWarp3(MobileUNetWarp):
    def __init__(self, target_size, n_classes, is_debug=False):
        self.warp_decoder.append(3)
        super(MobileUNetWarp3, self).__init__(target_size, n_classes, is_debug=is_debug)


class MobileUNetWarp2(MobileUNetWarp):
    def __init__(self, target_size, n_classes, is_debug=False):
        self.warp_decoder.append(2)
        super(MobileUNetWarp2, self).__init__(target_size, n_classes, is_debug=is_debug)


class MobileUNetWarp1(MobileUNetWarp):
    def __init__(self, target_size, n_classes, is_debug=False):
        self.warp_decoder.append(1)
        super(MobileUNetWarp1, self).__init__(target_size, n_classes, is_debug=is_debug)


class MobileUNetWarp24(MobileUNetWarp):
    def __init__(self, target_size, n_classes, is_debug=False):
        self.warp_decoder.append(2)
        self.warp_decoder.append(4)
        super(MobileUNetWarp24, self).__init__(target_size, n_classes, is_debug=is_debug)


class MobileUNetWarp124(MobileUNetWarp):
    def __init__(self, target_size, n_classes, is_debug=False):
        self.warp_decoder.append(1)
        self.warp_decoder.append(2)
        self.warp_decoder.append(4)
        super(MobileUNetWarp124, self).__init__(target_size, n_classes, is_debug=is_debug)


class MobileUNetWarp0(MobileUNetWarp):
    def __init__(self, target_size, n_classes, is_debug=False):
        self.warp_decoder.append(0)
        super(MobileUNetWarp0, self).__init__(target_size, n_classes, is_debug=is_debug)


if __name__ == '__main__':
    target_size = 256, 512
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = MobileUNetWarp2(target_size, 32)
    print(model.summary())
    keras.utils.plot_model(model.k, model.name + '.png', show_shapes=True, show_layer_names=True)
