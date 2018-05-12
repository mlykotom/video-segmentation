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

        transformed_flow = flow_cnn(self.target_size)(all_inputs)

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
            if self.training_phase:
                y_old = branch_quarter(z_old)
            else:
                input_branch_quarter = Input(branch_quarter.output_shape[1:], name='prev_branch_14')
                y_old = input_branch_quarter
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
            if self.training_phase:
                y_old_ = conv3_1_sub2_proj_bn(conv3_1_sub2_proj(z_old))
            else:
                input_branch_half = Input(conv3_1_sub2_proj_bn.output_shape[1:], name='prev_conv3_1_sub2_proj_bn')
                y_old_ = input_branch_half

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
            if self.training_phase:
                y_old = block_0(x_old)
            else:
                input_branch_full = Input(block_0.output_shape[1:], name='prev_branch_1')
                y_old = input_branch_full

            # y = netwarp(block_0.output_shape[1:])([y_old, y, transformed_flow])
            y = netwarp(y_old, y, transformed_flow)

        y = Add(name='sub12_sum')([y, y_])
        y = Activation('relu', name='sub12_sum/relu')(y)
        y = BilinearUpSampling2D(name='sub12_sum_interp')(y)

        if not self.training_phase:
            if 0 in self.warp_decoder:
                all_inputs.append(input_branch_full)
            if 1 in self.warp_decoder:
                all_inputs.append(input_branch_half)
            if 2 in self.warp_decoder:
                all_inputs.append(input_branch_quarter)

        return self.out_block(all_inputs, y, aux_1, aux_2)

    @staticmethod
    def get_custom_objects():
        custom_objects = ICNet.get_custom_objects()
        custom_objects.update({
            'Warp': Warp,
            'LinearCombination': LinearCombination
        })
        return custom_objects


class ICNetWarp0(ICNetWarp):
    def _prepare(self):
        self.warp_decoder.append(0)
        super(ICNetWarp0, self)._prepare()


class ICNetWarp1(ICNetWarp):
    def _prepare(self):
        self.warp_decoder.append(1)
        super(ICNetWarp1, self)._prepare()


class ICNetWarp2(ICNetWarp):
    def _prepare(self):
        self.warp_decoder.append(2)
        super(ICNetWarp2, self)._prepare()


class ICNetWarp01(ICNetWarp):
    def _prepare(self):
        self.warp_decoder.append(0)
        self.warp_decoder.append(1)
        super(ICNetWarp01, self)._prepare()


class ICNetWarp12(ICNetWarp):
    def _prepare(self):
        self.warp_decoder.append(1)
        self.warp_decoder.append(2)
        super(ICNetWarp12, self)._prepare()


class ICNetWarp012(ICNetWarp):
    def _prepare(self):
        self.warp_decoder.append(0)
        self.warp_decoder.append(1)
        self.warp_decoder.append(2)
        super(ICNetWarp012, self)._prepare()


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path

        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    else:
        __package__ = ''

    target_size = 256, 512
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = ICNetWarp0(target_size, 35, for_training=False)
    print(model.summary())
    model.save_json()
    model.plot_model()

    # model.k.load_weights('/home/mlyko/weights/city/rel/ICNetWarp12/0421:11e150.b8.lr=0.001000._dec=0.051000.of=farn.h5', by_name=True)
    # print("succeeded")
