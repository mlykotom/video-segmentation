import keras
from keras import Input
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Reshape, Lambda, \
    Add, concatenate, Conv2D, SpatialDropout2D
from keras.models import Model

from base_model import BaseModel
from layers import tf_warp


class SegNetWarpDiff(BaseModel):
    def __init__(self, target_size, n_classes, is_debug=False):
        self._filter_size = 64
        self._pool_size = (2, 2)
        self._kernel_size = (3, 3)

        super(SegNetWarpDiff, self).__init__(target_size, n_classes, is_debug)

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
        out = tf_warp(img, flow, out_size)
        return out

    def _block(self, input, filter_size, kernel_size, pool_size):
        out = Convolution2D(filter_size, kernel_size, padding='same')(input)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        if pool_size is not None:
            out = MaxPooling2D(pool_size=pool_size)(out)
        return out

    def _create_model(self):
        img_old = Input(shape=self.target_size + (3,), name='data_old')
        img_new = Input(shape=self.target_size + (3,), name='data_new')
        flo = Input(shape=self.target_size + (2,), name='data_flow')
        diff = Input(shape=self.target_size + (3,), name='data_diff')

        all_inputs = [img_old, img_new, flo, diff]

        # encoder
        transformed_flow = self.netwarp_module(img_old, img_new, flo, diff)

        flow1 = MaxPooling2D(pool_size=self._pool_size, name='flow_down_1')(transformed_flow)
        flow2 = MaxPooling2D(pool_size=self._pool_size, name='flow_down_2')(flow1)
        flow3 = MaxPooling2D(pool_size=self._pool_size, name='flow_down_3')(flow2)

        # new branch
        new_branch = self._block(img_new, self._filter_size, self._kernel_size, self._pool_size)
        old_branch = self._block(img_old, self._filter_size, self._kernel_size, self._pool_size)

        warped1 = Lambda(self.warp, name="warp1")([old_branch, flow1])
        out = Add()([warped1, new_branch])

        # warped1 = self._block(warped1, 128, kernel_size, pool_size)
        # warped1 = self._block(warped1, 256, kernel_size, pool_size)
        # warped1 = self._block(warped1, 512, kernel_size, pool_size=None)

        new_branch2 = self._block(out, 128, self._kernel_size, self._pool_size)
        # new_branch2 = self._block(new_branch, 128, kernel_size, pool_size)
        old_branch2 = self._block(old_branch, 128, self._kernel_size, self._pool_size)

        # warped2 = Lambda(self.warp, name="warp2")([old_branch2, flow2])
        # warped2 = self._block(warped2, 256, kernel_size, pool_size)
        # warped2 = self._block(warped2, 512, kernel_size, pool_size=None)

        new_branch3 = self._block(new_branch2, 256, self._kernel_size, self._pool_size)
        old_branch3 = self._block(old_branch2, 256, self._kernel_size, self._pool_size)

        new_branch4 = self._block(new_branch3, 512, self._kernel_size, pool_size=None)
        old_branch4 = self._block(old_branch3, 512, self._kernel_size, pool_size=None)

        out = new_branch4

        # warped3 = Lambda(self.warp, name="warp3")([old_branch4, flow3])
        # out = Add()([warped1, new_branch4])

        if not self.is_debug:
            out = SpatialDropout2D(0.3)(out)

        # decoder
        out = Convolution2D(512, self._kernel_size, padding='same')(out)
        out = BatchNormalization()(out)

        out = UpSampling2D(size=self._pool_size)(out)
        out = Convolution2D(256, self._kernel_size, padding='same')(out)
        out = BatchNormalization()(out)

        out = UpSampling2D(size=self._pool_size)(out)
        out = Convolution2D(128, self._kernel_size, padding='same')(out)
        out = BatchNormalization()(out)

        out = UpSampling2D(size=self._pool_size)(out)
        out = Convolution2D(self._filter_size, self._kernel_size, padding='same')(out)
        out = BatchNormalization()(out)

        out = Convolution2D(self.n_classes, (1, 1), padding='same')(out)

        out = Reshape((-1, self.n_classes))(out)
        out = Activation('softmax')(out)

        model = Model(inputs=all_inputs, outputs=[out])

        return model


if __name__ == '__main__':
    target_size = (288, 480)
    model = SegNetWarpDiff(target_size, 34)

    print(model.summary())
    keras.utils.plot_model(model.k, 'segnet_warp_diff_w1.png', show_shapes=True, show_layer_names=True)
