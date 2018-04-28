import keras
import keras.backend as K
import tensorflow as tf
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Input, Activation
from keras.constraints import Constraint
from keras.engine import Layer
from keras.layers import Lambda, Conv2D, concatenate, Subtract
import cv2


def get_layer_name(prefix):
    return prefix + '_' + str(K.get_uid(prefix))


class ResizeBilinear(Lambda):
    def __init__(self, out_size, **kwargs):
        def resize(image):
            return K.tf.image.resize_bilinear(image, out_size)

        super(ResizeBilinear, self).__init__(resize, **kwargs)


class MinMaxConstraint(Constraint):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max

    def __call__(self, w):
        return K.maximum(self.min, K.minimum(self.max, w))

    def get_config(self):
        config = {'min': self.min, 'max': self.max}
        base_config = super(MinMaxConstraint, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LinearCombination(Layer):
    def __init__(self, init_weights=[1., 1.], **kwargs):
        self.init_weights = init_weights
        super(LinearCombination, self).__init__(**kwargs)

    def get_config(self):
        config = {'init_weights': self.init_weights}
        base_config = super(LinearCombination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if input_shape[0] != input_shape[1]:
            raise Exception("Input shapes must be the same to LinearCombination layer")

        channels = input_shape[0][-1]

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
        return K.tf.multiply(self.w1, inputs[0]) + K.tf.multiply(self.w2, inputs[1])


class RGB2Gray(Lambda):
    def __init__(self, **kwargs):
        def convertor(input):
            return tf.image.rgb_to_grayscale(input)

        super(RGB2Gray, self).__init__(convertor, **kwargs)


class MorphOpeningDiffBW(Lambda):
    def __init__(self, **kwargs):
        ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel = tf.expand_dims(tf.convert_to_tensor(ellipse, dtype=tf.int32), axis=-1)

        self.strides = [1, 1, 1, 1]
        self.rates = [1, 1, 1, 1]
        super(MorphOpeningDiffBW, self).__init__(self._morph_open, **kwargs)

    def _morph_open(self, input_old_new):
        gray_old = tf.image.rgb_to_grayscale(input_old_new[0])
        gray_new = tf.image.rgb_to_grayscale(input_old_new[1])

        diff_tf = tf.subtract(gray_old, gray_new)
        diff_tf = tf.image.convert_image_dtype(diff_tf, dtype=tf.int32)

        erode = tf.nn.erosion2d(diff_tf, self.kernel, self.strides, self.rates, padding='SAME')
        dilate = tf.nn.dilation2d(erode, self.kernel, self.strides, self.rates, padding='SAME')
        dilate = tf.cast(dilate, tf.float32) / 256.0

        out = tf.clip_by_value(dilate, 0, 1)
        return out


def flow_cnn(img_old, img_new, flo):
    diff = Warp(name='img_diff')([img_old, flo])
    # opened = MorphOpeningDiffBW()([img_new, img_old])
    x = concatenate([flo, img_new, diff])  # opened
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.1))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.1))(x)
    x = Conv2D(2, (3, 3), padding='same', kernel_initializer=RandomNormal(stddev=0.1))(x)
    x = concatenate([flo, x])

    transformed_flow = Conv2D(
        2, (1, 1),
        # 2, (3, 3),  # TODO changed to 3,3
        padding='same',
        name='transformed_flow',
        kernel_initializer=RandomNormal(stddev=0.001),
    )(x)
    return transformed_flow


def netwarp(layer_old, layer_new, transformed_flow):
    out_size = layer_old.get_shape().as_list()[1:3]
    resized_flow = ResizeBilinear(out_size)(transformed_flow)

    warped = Warp()([layer_old, resized_flow])
    # combined = keras.layers.Add()([layer_new, warped])  # TODO changed !!
    combined = LinearCombination()([layer_new, warped])
    return combined


def netwarp_module(img_old, img_new, flo):
    diff = keras.layers.Subtract(name='img_diff')([img_old, img_new])

    x = concatenate([img_old, img_new, flo, diff])
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    x = concatenate([flo, x])
    # x = BatchNormalization()(x)
    x = Conv2D(2, (3, 3), padding='same', name='transformed_flow')(x)
    return x


class Warp(Layer):
    def __init__(self, resize=False, **kwargs):
        super(Warp, self).__init__(**kwargs)
        self.resize = resize

    def get_config(self):
        config = {'resize': self.resize}
        base_config = super(Warp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        img = inputs[0]
        flow = inputs[1]

        out_size = img.get_shape().as_list()[1:3]
        if self.resize:
            flow = ResizeBilinear(out_size)(flow)

        out = self.tf_warp(img, flow, out_size)
        return out

    @staticmethod
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


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

if __name__ == '__main__':
    target_size = 256, 512
    img_old = Input(target_size + (3,), name='img_old')
    k_layer = Input(target_size + (3,), name='k_layer_prev')
    img = Input(target_size + (3,), name='img_current')
    flo = Input(target_size + (2,), name='flow')

    x = flow_cnn(img_old, img, flo)
    model = Model([img_old, img, flo], x, name='FlowCNN')

    svg = SVG(model_to_dot(model, rankdir='LR', show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))

    with open('flow_cnn.svg', 'wb') as file:
        file.write(svg.data)
