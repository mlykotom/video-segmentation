from keras.layers import Activation, BatchNormalization
from keras.layers import Add
from keras.layers import ZeroPadding2D

from icnet import ICNet
from layers import BilinearUpSampling2D
from layers.warp import *


class ICNetWarp(ICNet):
    warp_decoder = []

    def _create_model(self):
        img_old = Input(shape=self.input_shape, name='data_old')
        img_new = Input(shape=self.input_shape, name='data_new')
        flo = Input(shape=self.target_size + (2,), name='data_flow')

        all_inputs = [img_old, img_new, flo]

        transformed_flow = flow_cnn(img_old, img_new, flo)

        x = img_new
        x_old = img_old

        # (1/2)
        branch_half = self.branch_half(self.input_shape)
        z = branch_half(x)
        z_old = branch_half(x_old)

        # (1/4)
        branch_quarter = self.branch_quarter(branch_half.output_shape[1:])
        y = branch_quarter(z)

        if 2 in self.warp_decoder:
            y_old = branch_quarter(z_old)
            # y = netwarp(branch_quarter.output_shape[1:])([y_old, y, transformed_flow])
            y = netwarp(y_old, y, transformed_flow)

        pyramid_block = self.pyramid_block(branch_quarter.output_shape[1:])
        aux_1 = pyramid_block(y)

        y = ZeroPadding2D(padding=2, name='padding17')(aux_1)
        y = Conv2D(128, 3, dilation_rate=2, name='conv_sub4')(y)
        y = BatchNormalization(name='conv_sub4_bn')(y)

        conv3_1_sub2_proj = Conv2D(128, 1, name='conv3_1_sub2_proj')
        conv3_1_sub2_proj_bn = BatchNormalization(name='conv3_1_sub2_proj_bn')

        y_ = conv3_1_sub2_proj_bn(conv3_1_sub2_proj(z))

        if 1 in self.warp_decoder:
            y_old_ = conv3_1_sub2_proj_bn(conv3_1_sub2_proj(z_old))
            # y_ = netwarp(conv3_1_sub2_proj_bn.output_shape[1:])([y_old_, y_, transformed_flow])
            y_ = netwarp(y_old_, y_, transformed_flow)

        y = Add(name='sub24_sum')([y, y_])
        y = Activation('relu', name='sub24_sum/relu')(y)

        aux_2 = BilinearUpSampling2D(name='sub24_sum_interp')(y)
        y = ZeroPadding2D(padding=2, name='padding18')(aux_2)
        y_ = Conv2D(128, 3, dilation_rate=2, name='conv_sub2')(y)
        y_ = BatchNormalization(name='conv_sub2_bn')(y_)

        # (1)
        block_0 = self.block_0(self.input_shape)
        y = block_0(x)

        if 0 in self.warp_decoder:
            y_old = block_0(x_old)
            # y = netwarp(block_0.output_shape[1:])([y_old, y, transformed_flow])
            y = netwarp(y_old, y, transformed_flow)

        y = Add(name='sub12_sum')([y, y_])
        y = Activation('relu', name='sub12_sum/relu')(y)
        y = BilinearUpSampling2D(name='sub12_sum_interp')(y)

        return self.out_block(all_inputs, y, aux_1, aux_2)

    @staticmethod
    def get_custom_objects():
        custom_objects = ICNet.get_custom_objects()
        custom_objects.update({
            'Warp': Warp,
            'ResizeBilinear': ResizeBilinear,
            'LinearCombination': LinearCombination
        })
        return custom_objects


class ICNetWarp0(ICNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0, for_training=True):
        self.warp_decoder.append(0)
        super(ICNetWarp0, self).__init__(target_size, n_classes, debug_samples=debug_samples, for_training=for_training)


class ICNetWarp1(ICNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0, for_training=True):
        self.warp_decoder.append(1)
        super(ICNetWarp1, self).__init__(target_size, n_classes, debug_samples=debug_samples, for_training=for_training)


class ICNetWarp2(ICNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0, for_training=True):
        self.warp_decoder.append(2)
        super(ICNetWarp2, self).__init__(target_size, n_classes, debug_samples=debug_samples, for_training=for_training)


class ICNetWarp01(ICNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0, for_training=True):
        self.warp_decoder.append(0)
        self.warp_decoder.append(1)
        super(ICNetWarp01, self).__init__(target_size, n_classes, debug_samples=debug_samples, for_training=for_training)


class ICNetWarp12(ICNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0, for_training=True):
        self.warp_decoder.append(1)
        self.warp_decoder.append(2)
        super(ICNetWarp12, self).__init__(target_size, n_classes, debug_samples=debug_samples, for_training=for_training)


class ICNetWarp012(ICNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0, for_training=True):
        self.warp_decoder.append(0)
        self.warp_decoder.append(1)
        self.warp_decoder.append(2)
        super(ICNetWarp012, self).__init__(target_size, n_classes, debug_samples=debug_samples, for_training=for_training)


if __name__ == '__main__':
    target_size = 256, 512
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = ICNetWarp2(target_size, 32)
    print(model.summary())
    model.plot_model()
