import keras
import keras.backend as K
import tensorflow as tf
from keras.engine import Layer
from keras.layers import Activation, BatchNormalization, concatenate
from keras.layers import Add
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import ZeroPadding2D

from icnet import ICNet
from layers import BilinearUpSampling2D
from layers import tf_warp


class FlowFilter(Layer):
    def __init__(self, init_value=1.0, **kwargs):
        self.init_value = init_value
        super(FlowFilter, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)

        self.fil = self.add_weight(
            name='filter',
            # shape=(1,),
            shape=input_shape,
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True,
            dtype='float32'
        )

        super(FlowFilter, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return K.tf.multiply(inputs, self.fil, name='flow_masked')
        # return K.tf.where(K.tf.greater(K.tf.abs(inputs), self.fil), inputs, inputs)


class Warp(Lambda):
    @staticmethod
    def _warp(x):
        img = x[0]
        flow = x[1]
        # flow = FlowFilter()(flow)

        out_size = img.get_shape().as_list()[1:3]
        resized_flow = Lambda(lambda image: tf.image.resize_bilinear(image, out_size))(flow)
        out = tf_warp(img, resized_flow, out_size)
        out = BatchNormalization()(out)
        return out

    def __init__(self, **kwargs):
        super(Warp, self).__init__(self._warp, **kwargs)


class ICNetWarp(ICNet):

    def netwarp_module(self, img_old, img_new, flo, diff):
        x = concatenate([img_old, img_new, flo, diff])
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
        x = concatenate([flo, x])
        x = BatchNormalization()(x)
        x = Conv2D(2, (3, 3), padding='same', name='transformed_flow')(x)
        return x

    def _create_model(self):
        img_old = Input(shape=self.target_size + (3,), name='data_old')
        img_new = Input(shape=self.target_size + (3,), name='data_new')
        flo = Input(shape=self.target_size + (2,), name='data_flow')
        diff = Input(shape=self.target_size + (3,), name='data_diff')

        all_inputs = [img_old, img_new, flo, diff]
        transformed_flow = self.netwarp_module(img_old, img_new, flo, diff)

        x = img_new

        # (1/2)
        z = self.branch_half(x)

        # (1/4)
        y = self.branch_quarter(z)

        aux_1 = self.pyramid_block(y)

        y = ZeroPadding2D(padding=2, name='padding17')(aux_1)
        y = Conv2D(128, 3, dilation_rate=2, name='conv_sub4')(y)
        y = BatchNormalization(name='conv_sub4_bn')(y)

        y_ = Conv2D(128, 1, name='conv3_1_sub2_proj')(z)
        y_ = BatchNormalization(name='conv3_1_sub2_proj_bn')(y_)

        y = Add(name='sub24_sum')([y, y_])
        y = Activation('relu', name='sub24_sum/relu')(y)

        aux_2 = BilinearUpSampling2D(name='sub24_sum_interp')(y)
        y = ZeroPadding2D(padding=2, name='padding18')(aux_2)
        y_ = Conv2D(128, 3, dilation_rate=2, name='conv_sub2')(y)
        y_ = BatchNormalization(name='conv_sub2_bn')(y_)

        # (1)
        y = self.block_0(x)
        y_old = self.block_0(img_old, prefix='old_')

        warped0 = Warp(name="warp1")([y_old, transformed_flow])

        y = Add(name='sub12_sum')([y, y_, warped0])
        y = Activation('relu', name='sub12_sum/relu')(y)
        y = BilinearUpSampling2D(name='sub12_sum_interp')(y)

        return self.out_block(all_inputs, y, aux_1, aux_2)


if __name__ == '__main__':
    target_size = 256, 512
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = ICNetWarp(target_size, 32)
    print(model.summary())
    model.plot_model()
