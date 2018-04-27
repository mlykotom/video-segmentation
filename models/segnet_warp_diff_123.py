from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Reshape, Add, SpatialDropout2D
from layers import *
from segnet_warp import SegNetWarp


class SegNetWarpDiff123(SegNetWarp):
    def _create_model(self):
        img_old = Input(shape=self.target_size + (3,), name='data_old')
        img_new = Input(shape=self.target_size + (3,), name='data_new')
        flo = Input(shape=self.target_size + (2,), name='data_flow')

        all_inputs = [img_old, img_new, flo]

        # encoder
        transformed_flow = netwarp_module(img_old, img_new, flo)

        flow1 = MaxPooling2D(pool_size=self._pool_size, name='flow_down_1')(transformed_flow)
        flow2 = MaxPooling2D(pool_size=self._pool_size, name='flow_down_2')(flow1)
        flow3 = MaxPooling2D(pool_size=self._pool_size, name='flow_down_3')(flow2)

        # new branch
        new_branch = self._block(img_new, self._filter_size, self._kernel_size, self._pool_size)
        old_branch = self._block(img_old, self._filter_size, self._kernel_size, self._pool_size)

        warped1 = Warp(name="warp1")([old_branch, flow1])
        warped1 = self._block(warped1, 128, self._kernel_size, self._pool_size)
        warped1 = self._block(warped1, 256, self._kernel_size, self._pool_size)
        warped1 = self._block(warped1, 512, self._kernel_size, pool_size=None)

        new_branch2 = self._block(new_branch, 128, self._kernel_size, self._pool_size)
        old_branch2 = self._block(old_branch, 128, self._kernel_size, self._pool_size)

        warped2 = Warp(name="warp2")([old_branch2, flow2])
        warped2 = self._block(warped2, 256, self._kernel_size, self._pool_size)
        warped2 = self._block(warped2, 512, self._kernel_size, pool_size=None)

        new_branch3 = self._block(new_branch2, 256, self._kernel_size, self._pool_size)
        old_branch3 = self._block(old_branch2, 256, self._kernel_size, self._pool_size)

        new_branch4 = self._block(new_branch3, 512, self._kernel_size, pool_size=None)
        old_branch4 = self._block(old_branch3, 512, self._kernel_size, pool_size=None)

        warped3 = Warp(name="warp3")([old_branch4, flow3])
        out = Add()([warped1, warped2, warped3, new_branch4])

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
    model = SegNetWarp(target_size, 34)

    print(model.summary())
    keras.utils.plot_model(model.k, 'segnet_warp_diff.png', show_shapes=True, show_layer_names=True)
