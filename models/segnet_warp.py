import keras
import tensorflow as tf
from keras import models, Input, Model
from keras.engine import Layer
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Reshape, \
    concatenate, Lambda

from base_model import BaseModel


class SegNet(BaseModel):

    def _create_model(self):
        model = models.Sequential()
        model.add(Layer(input_shape=(self.target_size[0], self.target_size[1], 3)))

        filter_size = 64
        pool_size = (2, 2)
        kernel_size = (3, 3)

        # encoder
        model.add(Convolution2D(filter_size, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))

        model.add(Convolution2D(128, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))

        model.add(Convolution2D(256, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))

        model.add(Convolution2D(512, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # decoder
        model.add(Convolution2D(512, kernel_size, padding='same'))
        model.add(BatchNormalization())

        model.add(UpSampling2D(size=pool_size))
        model.add(Convolution2D(256, kernel_size, padding='same'))
        model.add(BatchNormalization())

        model.add(UpSampling2D(size=pool_size))
        model.add(Convolution2D(128, kernel_size, padding='same'))
        model.add(BatchNormalization())

        model.add(UpSampling2D(size=pool_size))
        model.add(Convolution2D(filter_size, kernel_size, padding='same'))
        model.add(BatchNormalization())

        model.add(Convolution2D(self.n_classes, (1, 1), padding='same'))

        model.add(Reshape((-1, self.n_classes)))
        model.add(Activation('softmax'))

        return model


class SegNetWarp(BaseModel):

    def first_block(self, input, filter_size, kernel_size, pool_size):
        model = Convolution2D(filter_size, kernel_size, padding='same')(input)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPooling2D(pool_size=pool_size)(model)
        return model

    def tf_warp(self, img, flow, target_size):
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
        x = tf.expand_dims(x, 0)

        y = tf.expand_dims(y, 0)
        y = tf.expand_dims(y, 0)

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        grid = tf.concat([x, y], axis=1)
        #    print grid.shape
        flows = grid + flow
        max_y = tf.cast(H - 1, tf.int32)
        max_x = tf.cast(W - 1, tf.int32)
        zero = tf.zeros([], dtype=tf.int32)

        x = flows[:, 0, :, :]
        y = flows[:, 1, :, :]
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

    def _create_model(self):
        input_shape = self.target_size + (3,)

        filter_size = 64
        pool_size = (2, 2)
        kernel_size = (3, 3)

        img_old = Input(shape=input_shape, name='data_0')
        img_new = Input(shape=input_shape, name='data_1')

        flow_shape = (2,) + self.target_size
        flo = Input(shape=flow_shape, name='flow')

        all_inputs = [img_old, img_new, flo]

        def warp_test(x):
            img = x[0]
            flow = x[1]

            out_size = img.get_shape().as_list()[1:3]
            return self.tf_warp(img, flow, out_size)

        # encoder
        warped = Lambda(warp_test, name="warp")([img_old, flo])
        left_branch = self.first_block(warped, filter_size, kernel_size, pool_size)

        right_branch = self.first_block(img_new, filter_size, kernel_size, pool_size)

        model = concatenate([left_branch, right_branch])

        model = Convolution2D(128, kernel_size, padding='same')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPooling2D(pool_size=pool_size)(model)

        model = Convolution2D(256, kernel_size, padding='same')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPooling2D(pool_size=pool_size)(model)

        model = Convolution2D(512, kernel_size, padding='same')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)

        # decoder
        model = Convolution2D(512, kernel_size, padding='same')(model)
        model = BatchNormalization()(model)

        model = UpSampling2D(size=pool_size)(model)
        model = Convolution2D(256, kernel_size, padding='same')(model)
        model = BatchNormalization()(model)

        model = UpSampling2D(size=pool_size)(model)
        model = Convolution2D(128, kernel_size, padding='same')(model)
        model = BatchNormalization()(model)

        model = UpSampling2D(size=pool_size)(model)
        model = Convolution2D(filter_size, kernel_size, padding='same')(model)
        model = BatchNormalization()(model)

        model = Convolution2D(self.n_classes, (1, 1), padding='same')(model)

        model = Reshape((-1, self.n_classes))(model)
        model = Activation('softmax')(model)

        return Model(all_inputs, model)


if __name__ == '__main__':
    target_size = (288, 480)
    model = SegNetWarp(target_size, 30)

    print(model.summary())
    keras.utils.plot_model(model.k, 'segnet_warp.png', show_shapes=True, show_layer_names=True)
