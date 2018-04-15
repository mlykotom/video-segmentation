from keras.layers import Activation, BatchNormalization
from keras.layers import Add
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import ZeroPadding2D

from icnet import ICNet
from layers import BilinearUpSampling2D, Warp, netwarp_module


class ICNetWarp(ICNet):

    def _create_model(self):
        img_old = Input(shape=self.target_size + (3,), name='data_old')
        img_new = Input(shape=self.target_size + (3,), name='data_new')
        flo = Input(shape=self.target_size + (2,), name='data_flow')

        all_inputs = [img_old, img_new, flo]
        transformed_flow = netwarp_module(img_old, img_new, flo)

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

    def get_custom_objects(self):
        custom_objects = super(ICNetWarp, self).get_custom_objects()
        custom_objects.update({
            'Warp': Warp,
        })
        return custom_objects


if __name__ == '__main__':
    target_size = 256, 512
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = ICNetWarp(target_size, 32)
    print(model.summary())
    model.plot_model()
