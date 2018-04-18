import keras
import keras.backend as K
import tensorflow as tf
from keras import Input, Model
from keras.constraints import Constraint
from keras.engine import Layer
from keras.layers import Lambda, Conv2D, concatenate


class MinMaxConstraint(Constraint):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max

    def __call__(self, w):
        w = K.maximum(self.min, K.minimum(self.max, w))
        # w *= K.cast(K.greater_equal(w, self.min), K.floatx())
        # w *= K.cast(K.less_equal(w, self.max), K.floatx())
        return w


class LinearCombination(Layer):
    def __init__(self, init_weights=[1., 1.], **kwargs):
        self.init_weights = init_weights
        super(LinearCombination, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape[0] != input_shape[1]:
            raise Exception("Input shapes must be the same to LinearCombination layer")

        channels = input_shape[0][-1]

        # self.inp = K.constant(1.0)

        self.w1 = self.add_weight(
            name='w1',
            shape=(channels,),
            initializer=keras.initializers.Constant(self.init_weights[0]),
            trainable=True,
            constraint=MinMaxConstraint()
        )

        self.w2 = self.add_weight(
            name='w2',
            shape=(channels,),
            initializer=keras.initializers.Constant(self.init_weights[1]),
            trainable=True,
            constraint=MinMaxConstraint()
        )

        super(LinearCombination, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, **kwargs):
        # trim_w1 = K.tf.minimum(K.tf.maximum(0., self.w1), 1.)
        # trim_w2 = K.tf.minimum(K.tf.maximum(0., self.w2), 1.)

        return K.tf.multiply(self.w1, inputs[0]) + K.tf.multiply(self.w2, inputs[1])


# class FlowCNN(Model):

import cv2

import numpy as np
def netwarp_module(img_old, img_new, flo):
    img_old_gray = Lambda(lambda inp: tf.image.rgb_to_grayscale(inp))(img_old)
    img_new_gray = Lambda(lambda inp: tf.image.rgb_to_grayscale(inp))(img_new)

    # diff = keras.layers.Subtract(name='data_diff')([img_old, img_new])
    diff = keras.layers.Subtract(name='data_diff')([img_old_gray, img_new_gray])

    # TODO erode dilate
    # TODO learn the kernel
    # kernel = tf.Constant(np.ones((5,5),np.uint8))
    # tf.nn.erosion2d(tf.nn.dilation2d(diff, ),)

    x = concatenate([img_old, img_new, flo, diff])
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    x = concatenate([flo, x])
    # x = BatchNormalization()(x)
    x = Conv2D(2, (3, 3), padding='same', name='transformed_flow')(x)
    return x


class Warp(Lambda):
    @staticmethod
    def _warp(x):
        img = x[0]
        flow = x[1]
        # flow = FlowFilter()(flow)

        out_size = img.get_shape().as_list()[1:3]
        resized_flow = Lambda(lambda image: K.tf.image.resize_bilinear(image, out_size))(flow)
        out = tf_warp(img, resized_flow, out_size)
        # out = BatchNormalization()(out)
        return out

    def __init__(self, **kwargs):
        super(Warp, self).__init__(self._warp, **kwargs)


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


# class FlowFilter(Layer):
#     def __init__(self, init_value=1.0, **kwargs):
#         self.init_value = init_value
#         super(FlowFilter, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         print(input_shape)
#
#         self.fil = self.add_weight(
#             name='filter',
#             # shape=(1,),
#             shape=input_shape,
#             initializer=keras.initializers.Constant(self.init_value),
#             trainable=True,
#             dtype='float32'
#         )
#
#         super(FlowFilter, self).build(input_shape)
#
#     def call(self, inputs, **kwargs):
#         return K.tf.multiply(inputs, self.fil, name='flow_masked')
#         # return K.tf.where(K.tf.greater(K.tf.abs(inputs), self.fil), inputs, inputs)


def tf_warp(img, flow, target_size):
    # TODO read https://stackoverflow.com/questions/34902782/interpolated-sampling-of-points-in-an-image-with-tensorflow
    # TODO read https://github.com/tensorflow/models/blob/master/research/transformer/spatial_transformer.py

    H, W = target_size

    def get_pixel_value(img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.
        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*H*W, )
        - y: flattened tensor of shape (B*H*W, )
        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(img, indices)

    #    H = 256
    #    W = 256
    x, y = tf.meshgrid(tf.range(W), tf.range(H))
    x = tf.expand_dims(x, 0)
    x = tf.expand_dims(x, -1)

    y = tf.expand_dims(y, 0)
    y = tf.expand_dims(y, -1)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    grid = tf.concat([x, y], axis=-1)
    #    print grid.shape
    flows = grid + flow
    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    x = flows[:, :, :, 0]
    y = flows[:, :, :, 1]
    x0 = x
    y0 = y
    x0 = tf.cast(x0, tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(y0, tf.int32)
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    return out
