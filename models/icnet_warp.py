from keras.layers import Activation, BatchNormalization, Conv1D
from keras.layers import Add
from keras.layers import Input
from keras.layers import ZeroPadding2D

from icnet import ICNet
from layers import BilinearUpSampling2D
from layers.warp import *


class ICNetWarp(ICNet):
    warp_decoder = []

    def _create_model(self):
        img_old = Input(shape=self.target_size + (3,), name='data_old')
        img_new = Input(shape=self.target_size + (3,), name='data_new')
        flo = Input(shape=self.target_size + (2,), name='data_flow')

        all_inputs = [img_old, img_new, flo]
        transformed_flow = netwarp_module(img_old, img_new, flo)

        x = img_new
        x_old = img_old

        # (1/2)
        z = self.branch_half(x)
        z_old = self.branch_half(x_old, prefix='old_')

        # (1/4)
        y = self.branch_quarter(z)

        if 2 in self.warp_decoder:
            y_old = self.branch_quarter(z_old, prefix='old_')
            warped2 = Warp(name="warp_2")([y_old, transformed_flow])
            y = LinearCombination()([y, warped2])

        aux_1 = self.pyramid_block(y)

        y = ZeroPadding2D(padding=2, name='padding17')(aux_1)
        y = Conv2D(128, 3, dilation_rate=2, name='conv_sub4')(y)
        y = BatchNormalization(name='conv_sub4_bn')(y)

        y_ = Conv2D(128, 1, name='old_conv3_1_sub2_proj')(z)
        y_ = BatchNormalization(name='conv3_1_sub2_proj_bn')(y_)

        if 1 in self.warp_decoder:
            y_old_ = Conv2D(128, 1, name='old_old_conv3_1_sub2_proj')(z_old)
            y_old_ = BatchNormalization(name='old_old_conv3_1_sub2_proj_bn')(y_old_)
            warped1 = Warp(name="warp_0")([y_old_, transformed_flow])
            y_ = LinearCombination()([y_, warped1])

        y = Add(name='sub24_sum')([y, y_])
        y = Activation('relu', name='sub24_sum/relu')(y)

        aux_2 = BilinearUpSampling2D(name='sub24_sum_interp')(y)
        y = ZeroPadding2D(padding=2, name='padding18')(aux_2)
        y_ = Conv2D(128, 3, dilation_rate=2, name='conv_sub2')(y)
        y_ = BatchNormalization(name='conv_sub2_bn')(y_)

        # (1)
        y = self.block_0(x)

        if 0 in self.warp_decoder:
            y_old = self.block_0(x_old, prefix='old_')
            warped0 = Warp(name="warp_1")([y_old, transformed_flow])
            y = LinearCombination()([y, warped0])

        y = Add(name='sub12_sum')([y, y_])
        y = Activation('relu', name='sub12_sum/relu')(y)
        y = BilinearUpSampling2D(name='sub12_sum_interp')(y)

        return self.out_block(all_inputs, y, aux_1, aux_2)

    def get_custom_objects(self):
        custom_objects = super(ICNetWarp, self).get_custom_objects()
        custom_objects.update({
            'Warp': Warp,
        })
        return custom_objects


class ICNetWarp0(ICNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0):
        self.warp_decoder.append(0)
        super(ICNetWarp0, self).__init__(target_size, n_classes, debug_samples=debug_samples)


class ICNetWarp1(ICNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0):
        self.warp_decoder.append(1)
        super(ICNetWarp1, self).__init__(target_size, n_classes, debug_samples=debug_samples)

    def optimizer_params(self):
        if self.debug_samples == 120:
            return {'lr': 0.0007, 'decay': 0.5}
            # return {'lr': 0.0002, 'decay': 0.0991}  # for 120 samples
        else:
            return super(ICNetWarp, self).optimizer_params()


class ICNetWarp2(ICNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0):
        self.warp_decoder.append(2)
        super(ICNetWarp2, self).__init__(target_size, n_classes, debug_samples=debug_samples)


class ICNetWarp01(ICNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0):
        self.warp_decoder.append(0)
        self.warp_decoder.append(1)
        super(ICNetWarp01, self).__init__(target_size, n_classes, debug_samples=debug_samples)

    def optimizer_params(self):
        if self.debug_samples == 120:
            return {'lr': 0.00065, 'decay': 0.35}
        else:
            return super(ICNetWarp, self).optimizer_params()

if __name__ == '__main__':
    target_size = 256, 512
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = ICNetWarp01(target_size, 32)
    print(model.summary())
    model.plot_model()
