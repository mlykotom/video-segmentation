from keras.layers import Convolution2D, BatchNormalization, UpSampling2D

from base_model import BaseModel
from layers import *
from segnet import SegNet


class SegNetWarp(SegNet):
    warp_decoder = []

    def _create_model(self):
        img_old = Input(self.input_shape, name='data_old')
        img_new = Input(self.input_shape, name='data_new')
        flo = Input(shape=self.target_size + (2,), name='data_flow')

        all_inputs = [img_old, img_new, flo]
        transformed_flow = flow_cnn(self.target_size)(all_inputs)

        # encoder
        block_0 = self.block_model(self.input_shape, 64, 1, True)
        block_1 = self.block_model(block_0.output_shape[1:], 128, 2, True)
        block_2 = self.block_model(block_1.output_shape[1:], 256, 3, True)
        block_3 = self.block_model(block_2.output_shape[1:], 512, 4, False)

        out = block_0(img_new)
        block_0_out = out
        old_out = block_0(img_old)

        if 0 in self.warp_decoder:
            if not self.training_phase:
                input_block_0 = Input(block_0.output_shape[1:], name='prev_conv_block_1')
                old_branch = input_block_0
            else:
                old_branch = old_out

            out = netwarp(old_branch, out, transformed_flow)

        out = block_1(out)
        block_1_out = out
        old_out = block_1(old_out)

        if 1 in self.warp_decoder:
            if not self.training_phase:
                input_block_1 = Input(block_1.output_shape[1:], name='prev_conv_block_2')
                old_branch = input_block_1
            else:
                old_branch = old_out

            out = netwarp(old_branch, out, transformed_flow)

        out = block_2(out)
        block_2_out = out
        old_out = block_2(old_out)

        if 2 in self.warp_decoder:
            if not self.training_phase:
                input_block_2 = Input(block_2.output_shape[1:], name='prev_conv_block_3')
                old_branch = input_block_2
            else:
                old_branch = old_out

            out = netwarp(old_branch, out, transformed_flow)

        out = block_3(out)
        block_3_out = out
        old_out = block_3(old_out)

        if 3 in self.warp_decoder:
            if not self.training_phase:
                input_block_3 = Input(block_3.output_shape[1:], name='prev_conv_block_4')
                old_branch = input_block_3
            else:
                old_branch = old_out

            out = netwarp(old_branch, out, transformed_flow)

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

        out = Convolution2D(self.n_classes, (1, 1), activation='softmax', padding='same')(out)

        outputs = [out]

        if not self.training_phase:
            if 0 in self.warp_decoder:
                all_inputs.append(input_block_0)
                outputs.append(block_0_out)
            if 1 in self.warp_decoder:
                all_inputs.append(input_block_1)
                outputs.append(block_1_out)
            if 2 in self.warp_decoder:
                all_inputs.append(input_block_2)
                outputs.append(block_2_out)
            if 3 in self.warp_decoder:
                all_inputs.append(input_block_3)
                outputs.append(block_3_out)

        model = Model(inputs=all_inputs, outputs=outputs)
        return model

    @staticmethod
    def get_custom_objects():
        custom_objects = BaseModel.get_custom_objects()
        custom_objects.update({
            'Warp': Warp,
            'ResizeBilinear': ResizeBilinear,
            'LinearCombination': LinearCombination
        })
        return custom_objects


class SegnetWarp0(SegNetWarp):
    def _prepare(self):
        self.warp_decoder.append(0)
        super(SegnetWarp0, self)._prepare()


class SegnetWarp1(SegNetWarp):
    def _prepare(self):
        self.warp_decoder.append(1)
        super(SegnetWarp1, self)._prepare()


class SegnetWarp2(SegNetWarp):
    def _prepare(self):
        self.warp_decoder.append(2)
        super(SegnetWarp2, self)._prepare()


class SegnetWarp3(SegNetWarp):
    def _prepare(self):
        self.warp_decoder.append(3)
        super(SegnetWarp3, self)._prepare()


class SegnetWarp01(SegNetWarp):
    def _prepare(self):
        self.warp_decoder.append(0)
        self.warp_decoder.append(1)
        super(SegnetWarp01, self)._prepare()


class SegnetWarp12(SegNetWarp):
    def _prepare(self):
        self.warp_decoder.append(1)
        self.warp_decoder.append(2)
        super(SegnetWarp12, self)._prepare()


class SegnetWarp23(SegNetWarp):
    def _prepare(self):
        self.warp_decoder.append(2)
        self.warp_decoder.append(3)
        super(SegnetWarp23, self)._prepare()


class SegnetWarp012(SegNetWarp):
    def _prepare(self):
        self.warp_decoder.append(0)
        self.warp_decoder.append(1)
        self.warp_decoder.append(2)
        super(SegnetWarp012, self)._prepare()


class SegnetWarp123(SegNetWarp):
    def _prepare(self):
        self.warp_decoder.append(1)
        self.warp_decoder.append(2)
        self.warp_decoder.append(3)
        super(SegnetWarp123, self)._prepare()


class SegnetWarp0123(SegNetWarp):
    def _prepare(self):
        self.warp_decoder.append(0)
        self.warp_decoder.append(1)
        self.warp_decoder.append(2)
        self.warp_decoder.append(3)
        super(SegnetWarp0123, self)._prepare()


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path

        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    else:
        __package__ = ''

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(os.getcwd())

    target_size = 256, 512
    model = SegnetWarp3(target_size, 35, for_training=False)
    print(model.summary())
