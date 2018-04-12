import keras
from keras import Input
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Reshape, Lambda, \
    Add
from keras.models import Model

from base_model import BaseModel
from layers import tf_warp


class SegNetWarp(BaseModel):

    def _block(self, input, filter_size, kernel_size, pool_size):
        out = Convolution2D(filter_size, kernel_size, padding='same')(input)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        if pool_size is not None:
            out = MaxPooling2D(pool_size=pool_size)(out)
        return out

    def _create_model(self):
        filter_size = 64
        pool_size = (2, 2)
        kernel_size = (3, 3)

        img_old = Input(shape=self.target_size + (3,), name='data_old')
        img_new = Input(shape=self.target_size + (3,), name='data_new')
        flo = Input(shape=self.target_size + (2,), name='flow')

        all_inputs = [img_old, img_new, flo]

        def warp_test(x):
            img = x[0]
            flow = x[1]

            out_size = img.get_shape().as_list()[1:3]
            flow_size = flow.get_shape().as_list()[1:3]

            # print(out_size, flow_size, flow_size[0] / out_size[0] / pool_size[0])

            if out_size < flow_size:
                how_many = flow_size[0] / out_size[0] / pool_size[0]
                for _ in range(1, how_many):
                    flow = MaxPooling2D(pool_size=pool_size)(flow)

            out = tf_warp(img, flow, out_size)
            return out

        def warp(x):
            img = x[0]
            flow = x[1]
            out_size = img.get_shape().as_list()[1:3]
            out = tf_warp(img, flow, out_size)
            return out

        # encoder
        flow1 = MaxPooling2D(pool_size=pool_size, name='flow_down_1')(flo)
        flow2 = MaxPooling2D(pool_size=pool_size, name='flow_down_2')(flow1)
        flow3 = MaxPooling2D(pool_size=pool_size, name='flow_down_3')(flow2)

        # new branch
        new_branch = self._block(img_new, filter_size, kernel_size, pool_size)
        old_branch = self._block(img_old, filter_size, kernel_size, pool_size)

        warped2 = Lambda(warp, name="warp1")([old_branch, flow1])
        warped2 = self._block(warped2, 128, kernel_size, pool_size)
        warped2 = self._block(warped2, 256, kernel_size, pool_size)
        warped2 = self._block(warped2, 512, kernel_size, pool_size=None)

        new_branch2 = self._block(new_branch, 128, kernel_size, pool_size)
        old_branch2 = self._block(old_branch, 128, kernel_size, pool_size)

        warped3 = Lambda(warp, name="warp2")([old_branch2, flow2])
        warped3 = self._block(warped3, 256, kernel_size, pool_size)
        warped3 = self._block(warped3, 512, kernel_size, pool_size=None)

        new_branch3 = self._block(new_branch2, 256, kernel_size, pool_size)
        old_branch3 = self._block(old_branch2, 256, kernel_size, pool_size)

        new_branch4 = self._block(new_branch3, 512, kernel_size, pool_size=None)
        old_branch4 = self._block(old_branch3, 512, kernel_size, pool_size=None)

        warped4 = Lambda(warp, name="warp3")([old_branch4, flow3])
        out = Add()([warped2, warped3, warped4, new_branch4])

        # warped = Lambda(warp_test, name="warp")([old_branch, flo])
        # out = concatenate([warped, new_branch])

        # out = new_branch

        # decoder
        out = Convolution2D(512, kernel_size, padding='same')(out)
        out = BatchNormalization()(out)

        out = UpSampling2D(size=pool_size)(out)
        out = Convolution2D(256, kernel_size, padding='same')(out)
        out = BatchNormalization()(out)

        out = UpSampling2D(size=pool_size)(out)
        out = Convolution2D(128, kernel_size, padding='same')(out)
        out = BatchNormalization()(out)

        out = UpSampling2D(size=pool_size)(out)
        out = Convolution2D(filter_size, kernel_size, padding='same')(out)
        out = BatchNormalization()(out)

        out = Convolution2D(self.n_classes, (1, 1), padding='same')(out)

        out = Reshape((-1, self.n_classes))(out)
        out = Activation('softmax')(out)

        model = Model(inputs=all_inputs, outputs=[out])

        return model


if __name__ == '__main__':
    target_size = (288, 480)
    model = SegNetWarp(target_size, 32)

    # print(model.summary())
    keras.utils.plot_model(model.k, 'segnet_warp.png', show_shapes=True, show_layer_names=True)
