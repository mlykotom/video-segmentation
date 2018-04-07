import keras
import keras.backend as K
from keras import Input, optimizers
from keras.applications import mobilenet
from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import Conv2D, BatchNormalization, Activation, concatenate, Conv2DTranspose, Reshape, Dropout, \
    MaxPooling2D, Lambda, SpatialDropout2D, Add
from keras.models import Model

from base_model import BaseModel
from layers import BilinearUpSampling2D

from layers import tf_warp

import tensorflow as tf


# TODO Mobilenet v2? https://github.com/xiaochus/MobileNetV2/blob/master/mobilenet_v2.py

class MobileUNetWarp(BaseModel):

    def __init__(self, target_size, n_classes, alpha=1.0, alpha_up=1.0, depth_multiplier=1, dropout=1e-3,
                 is_debug=False):
        self.alpha = alpha
        self.alpha_up = alpha_up
        self.depth_multiplier = depth_multiplier
        self.dropout = dropout

        super(MobileUNetWarp, self).__init__(target_size, n_classes, is_debug)

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
        transformed_flow = Conv2D(2, (3, 3), padding='same', name='transformed_flow')(x)
        return transformed_flow

    def warp(self, x):
        img = x[0]
        flow = x[1]
        out_size = img.get_shape().as_list()[1:3]
        flow = Lambda(lambda image: tf.image.resize_bilinear(image, out_size))(flow)
        out = tf_warp(img, flow, out_size)
        return out

    def _create_model(self):
        img_old = Input(shape=self.target_size + (3,), name='data_old')
        img_new = Input(shape=self.target_size + (3,), name='data_new')
        flo = Input(shape=self.target_size + (2,), name='data_flow')
        diff = Input(shape=self.target_size + (3,), name='data_diff')

        input = img_new
        all_inputs = [img_old, img_new, flo, diff]
        transformed_flow = self.netwarp_module(img_old, img_new, flo, diff)

        new_b00 = self._conv_block(input, 32, self.alpha, strides=(2, 2), block_id=0)
        new_b01 = self._depthwise_conv_block(new_b00, 64, self.alpha, self.depth_multiplier, block_id=1)
        new_b02 = self._depthwise_conv_block(new_b01, 128, self.alpha, self.depth_multiplier, block_id=2, strides=(2, 2))
        new_b03 = self._depthwise_conv_block(new_b02, 128, self.alpha, self.depth_multiplier, block_id=3)
        new_b04 = self._depthwise_conv_block(new_b03, 256, self.alpha, self.depth_multiplier, block_id=4, strides=(2, 2))
        new_b05 = self._depthwise_conv_block(new_b04, 256, self.alpha, self.depth_multiplier, block_id=5)
        new_b06 = self._depthwise_conv_block(new_b05, 512, self.alpha, self.depth_multiplier, block_id=6, strides=(2, 2))
        new_b07 = self._depthwise_conv_block(new_b06, 512, self.alpha, self.depth_multiplier, block_id=7)
        new_b08 = self._depthwise_conv_block(new_b07, 512, self.alpha, self.depth_multiplier, block_id=8)
        new_b09 = self._depthwise_conv_block(new_b08, 512, self.alpha, self.depth_multiplier, block_id=9)
        new_b10 = self._depthwise_conv_block(new_b09, 512, self.alpha, self.depth_multiplier, block_id=10)
        new_b11 = self._depthwise_conv_block(new_b10, 512, self.alpha, self.depth_multiplier, block_id=11)
        new_b12 = self._depthwise_conv_block(new_b11, 1024, self.alpha, self.depth_multiplier, block_id=12, strides=(2, 2))
        new_b13 = self._depthwise_conv_block(new_b12, 1024, self.alpha, self.depth_multiplier, block_id=13)

        old_b00 = self._conv_block(img_old, 32, self.alpha, strides=(2, 2), block_id=0, prefix='old_')
        old_b01 = self._depthwise_conv_block(old_b00, 64, self.alpha, self.depth_multiplier, block_id=1, prefix='old_')
        # # --
        # old_b02 = self._depthwise_conv_block(old_b01, 128, self.alpha, self.depth_multiplier, block_id=2, strides=(2, 2), prefix='old_')
        # old_b03 = self._depthwise_conv_block(old_b02, 128, self.alpha, self.depth_multiplier, block_id=3, prefix='old_')
        # # --
        # old_b04 = self._depthwise_conv_block(old_b03, 256, self.alpha, self.depth_multiplier, block_id=4, strides=(2, 2), prefix='old_')
        # old_b05 = self._depthwise_conv_block(old_b04, 256, self.alpha, self.depth_multiplier, block_id=5, prefix='old_')
        # # --
        # old_b06 = self._depthwise_conv_block(old_b05, 512, self.alpha, self.depth_multiplier, block_id=6, strides=(2, 2), prefix='old_')
        # old_b07 = self._depthwise_conv_block(old_b06, 512, self.alpha, self.depth_multiplier, block_id=7, prefix='old_')
        # old_b08 = self._depthwise_conv_block(old_b07, 512, self.alpha, self.depth_multiplier, block_id=8, prefix='old_')
        # old_b09 = self._depthwise_conv_block(old_b08, 512, self.alpha, self.depth_multiplier, block_id=9, prefix='old_')
        # old_b10 = self._depthwise_conv_block(old_b09, 512, self.alpha, self.depth_multiplier, block_id=10, prefix='old_')
        # old_b11 = self._depthwise_conv_block(old_b10, 512, self.alpha, self.depth_multiplier, block_id=11, prefix='old_')
        # # --
        # old_b12 = self._depthwise_conv_block(old_b11, 1024, self.alpha, self.depth_multiplier, block_id=12, strides=(2, 2), prefix='old_')
        # old_b13 = self._depthwise_conv_block(old_b12, 1024, self.alpha, self.depth_multiplier, block_id=13, prefix='old_')

        # flow1 = MaxPooling2D(pool_size=(2, 2), name='flow_down_1')(transformed_flow)
        # flow2 = MaxPooling2D(pool_size=(2, 2), name='flow_down_2')(flow1)
        # flow3 = MaxPooling2D(pool_size=(2, 2), name='flow_down_3')(flow2)
        # flow4 = MaxPooling2D(pool_size=(2, 2), name='flow_down_4')(flow3)
        #
        # if not self.is_debug:
        #     old_b01 = SpatialDropout2D(0.1)(old_b01)

        # warped0 = Lambda(self.warp, name="warp0")([old_b00, transformed_flow])
        # warped0_after = self._depthwise_conv_block(warped0, 64, self.alpha, self.depth_multiplier, block_id=1, prefix='old_')
        # if not self.is_debug:
        #     warped0_after = SpatialDropout2D(0.2)(warped0_after)

        if not self.is_debug:
            old_b01 = SpatialDropout2D(0.4)(old_b01)

        warped1 = Lambda(self.warp, name="warp1")([old_b01, transformed_flow])
        # warped2 = Lambda(self.warp, name="warp2")([old_b03, flow2])
        # warped3 = Lambda(self.warp, name="warp3")([old_b05, flow3])
        # warped4 = Lambda(self.warp, name="warp3")([old_b11, flow4])

        if not self.is_debug:
            new_b13 = SpatialDropout2D(0.4)(new_b13)

        filters = int(512 * self.alpha)
        up1 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(new_b13),
            new_b11,
            # warped4
        ], axis=3)
        # up1 = BatchNormalization()(up1)

        b14 = self._depthwise_conv_block(up1, filters, self.alpha_up, self.depth_multiplier, block_id=14)

        filters = int(256 * self.alpha)
        up2 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b14),
            new_b05,
            # warped3
        ], axis=3)
        # up2 = BatchNormalization()(up2)

        b15 = self._depthwise_conv_block(up2, filters, self.alpha_up, self.depth_multiplier, block_id=15)

        filters = int(128 * self.alpha)
        up3 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b15),
            new_b03,
            # warped2
        ], axis=3)
        # up3 = BatchNormalization()(up3)

        b16 = self._depthwise_conv_block(up3, filters, self.alpha_up, self.depth_multiplier, block_id=16)

        up4 = Add()([warped1, new_b01])

        filters = int(64 * self.alpha)
        up4 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b16),
            up4
            # new_b01,
        ], axis=3)
        # up4 = BatchNormalization()(up4)

        b17 = self._depthwise_conv_block(up4, filters, self.alpha_up, self.depth_multiplier, block_id=17)

        filters = int(32 * self.alpha)
        # up5 = concatenate([
        #     b17,
        #     new_b00,
        #     # warped1
        #     # warped0_after
        # ], axis=3)

        up5 = concatenate([
            b17,
            new_b00,
        ], axis=3)

        # up5 = BatchNormalization()(up5)

        b18 = self._depthwise_conv_block(up5, filters, self.alpha_up, self.depth_multiplier, block_id=18)
        b18 = self._conv_block(b18, filters, self.alpha_up, block_id=18)

        x = Conv2D(self.n_classes, (1, 1), kernel_initializer='he_normal', activation='linear')(b18)
        x = BilinearUpSampling2D(size=(2, 2))(x)

        x = Reshape((-1, self.n_classes))(x)
        x = Activation('softmax')(x)

        return Model(all_inputs, x)

    custom_objects = {
        'relu6': mobilenet.relu6,
        'DepthwiseConv2D': mobilenet.DepthwiseConv2D,
        'BilinearUpSampling2D': BilinearUpSampling2D,
    }

    def _compile_release(self, m_metrics):
        self._model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=optimizers.Adam(decay=0.0001),
            metrics=m_metrics
        )


if __name__ == '__main__':
    target_size = 256, 512
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = MobileUNetWarp(target_size, 32)
    # model.k.predict()
    print(model.summary())
    keras.utils.plot_model(model.k, 'mobile_unet_warp1_add_bn+up5.png', show_shapes=True, show_layer_names=True)
