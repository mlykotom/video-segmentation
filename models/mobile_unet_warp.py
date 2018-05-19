from keras.layers import Activation, Conv2DTranspose

from layers import BilinearUpSampling2D
from layers.warp import *
from mobile_unet import MobileUNet


class MobileUNetWarp(MobileUNet):
    """
    WARNING: uses 2 branches for geting current segmentation mask instead of one with shared weights!
    This was old way of experimenting with the model, so it's not updated to latest way of doing video segmentation.
    If you want to use MobileUnet with video warping, needs to be updated based on ICNetWarp or SegNetWarp
    """

    warp_decoder = []

    def __init__(self, target_size, n_classes, alpha=1.0, alpha_up=1.0, depth_multiplier=1, dropout=1e-3, debug_samples=0):
        self.alpha = alpha
        self.alpha_up = alpha_up
        self.depth_multiplier = depth_multiplier
        self.dropout = dropout
        super(MobileUNetWarp, self).__init__(target_size, n_classes, debug_samples=debug_samples)

    def frame_branch(self, img_input, prefix=''):
        b00 = self._conv_block(img_input, 32, self.alpha, strides=(2, 2), block_id=0, prefix=prefix)
        b01 = self._depthwise_conv_block(b00, 64, self.alpha, self.depth_multiplier, block_id=1, prefix=prefix)
        # --
        b02 = self._depthwise_conv_block(b01, 128, self.alpha, self.depth_multiplier, block_id=2, strides=(2, 2), prefix=prefix)
        b03 = self._depthwise_conv_block(b02, 128, self.alpha, self.depth_multiplier, block_id=3, prefix=prefix)
        # --
        b04 = self._depthwise_conv_block(b03, 256, self.alpha, self.depth_multiplier, block_id=4, strides=(2, 2), prefix=prefix)
        b05 = self._depthwise_conv_block(b04, 256, self.alpha, self.depth_multiplier, block_id=5, prefix=prefix)
        # --
        b06 = self._depthwise_conv_block(b05, 512, self.alpha, self.depth_multiplier, block_id=6, strides=(2, 2), prefix=prefix)
        b07 = self._depthwise_conv_block(b06, 512, self.alpha, self.depth_multiplier, block_id=7, prefix=prefix)
        b08 = self._depthwise_conv_block(b07, 512, self.alpha, self.depth_multiplier, block_id=8, prefix=prefix)
        b09 = self._depthwise_conv_block(b08, 512, self.alpha, self.depth_multiplier, block_id=9, prefix=prefix)
        b10 = self._depthwise_conv_block(b09, 512, self.alpha, self.depth_multiplier, block_id=10, prefix=prefix)
        b11 = self._depthwise_conv_block(b10, 512, self.alpha, self.depth_multiplier, block_id=11, prefix=prefix)
        # --
        b12 = self._depthwise_conv_block(b11, 1024, self.alpha, self.depth_multiplier, block_id=12, strides=(2, 2), prefix=prefix)
        b13 = self._depthwise_conv_block(b12, 1024, self.alpha, self.depth_multiplier, block_id=13, prefix=prefix)

        return b00, b01, b03, b05, b11, b13

    def decoder(self):
        filters = int(512 * self.alpha)

        # b11 = BatchNormalization()(Add()([self.b11, self.warped4])) if 4 in self.warp_decoder else self.b11
        b11 = LinearCombination()([self.b11, self.warped4]) if 4 in self.warp_decoder else self.b11

        up1 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(self.b13),
            b11,
        ], axis=3)

        b14 = self._depthwise_conv_block(up1, filters, self.alpha_up, self.depth_multiplier, block_id=14)

        filters = int(256 * self.alpha)

        b05 = LinearCombination()([self.b05, self.warped3]) if 3 in self.warp_decoder else self.b05

        up2 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b14),
            b05,
        ], axis=3)

        b15 = self._depthwise_conv_block(up2, filters, self.alpha_up, self.depth_multiplier, block_id=15)

        filters = int(128 * self.alpha)

        b03 = LinearCombination()([self.b03, self.warped2]) if 2 in self.warp_decoder else self.b03

        up3 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b15),
            b03,
        ], axis=3)

        b16 = self._depthwise_conv_block(up3, filters, self.alpha_up, self.depth_multiplier, block_id=16)

        filters = int(64 * self.alpha)

        b01 = LinearCombination()([self.b01, self.warped1]) if 1 in self.warp_decoder else self.b01

        up4 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b16),
            b01
        ], axis=3)

        b17 = self._depthwise_conv_block(up4, filters, self.alpha_up, self.depth_multiplier, block_id=17)

        filters = int(32 * self.alpha)

        b00 = LinearCombination()([self.b00, self.warped0]) if 0 in self.warp_decoder else self.b00

        up5 = concatenate([
            b17,
            b00
        ], axis=3)

        b18 = self._depthwise_conv_block(up5, filters, self.alpha_up, self.depth_multiplier, block_id=18)
        b18 = self._conv_block(b18, filters, self.alpha_up, block_id=18)

        return b18

    def _create_model(self):
        img_old = Input(shape=self.target_size + (3,), name='data_old')
        img_new = Input(shape=self.target_size + (3,), name='data_new')
        flo = Input(shape=self.target_size + (2,), name='data_flow')

        all_inputs = [img_old, img_new, flo]
        transformed_flow = flow_cnn(self.target_size)(all_inputs)

        # -------- OLD FRAME BRANCH
        self.old_b00, self.old_b01, self.old_b03, self.old_b05, self.old_b11, self.old_b13 = self.frame_branch(img_old, prefix='old_')

        # -------- ACTUAL FRAME BRANCH
        self.b00, self.b01, self.b03, self.b05, self.b11, self.b13 = self.frame_branch(img_new)

        # -------- WARPING
        if -1 in self.warp_decoder:
            self.warped_inp = Warp(name="warp_inp")([img_old, transformed_flow])
        if 0 in self.warp_decoder:
            self.warped0 = Warp(name="warp0")([self.old_b00, transformed_flow])
        if 1 in self.warp_decoder:
            self.warped1 = Warp(name="warp1")([self.old_b01, transformed_flow])
        if 2 in self.warp_decoder:
            self.warped2 = Warp(name="warp2")([self.old_b03, transformed_flow])
        if 3 in self.warp_decoder:
            self.warped3 = Warp(name="warp3")([self.old_b05, transformed_flow])
        if 4 in self.warp_decoder:
            self.warped4 = Warp(name="warp4")([self.old_b11, transformed_flow])
        if 5 in self.warp_decoder:
            self.warped5 = Warp(name="warp5")([self.old_b13, transformed_flow])

        # -------- DECODER
        x = self.decoder()

        x = Conv2D(self.n_classes, (1, 1), kernel_initializer='he_normal', activation='linear')(x)
        x = BilinearUpSampling2D()(x)
        x = Activation('softmax')(x)

        return Model(all_inputs, x)

    @staticmethod
    def get_custom_objects():
        custom_objects = MobileUNet.get_custom_objects()
        custom_objects.update({
            'Warp': Warp
        })
        return custom_objects


class MobileUNetWarp4(MobileUNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0):
        self.warp_decoder.append(4)
        super(MobileUNetWarp4, self).__init__(target_size, n_classes, debug_samples=debug_samples)


class MobileUNetWarp3(MobileUNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0):
        self.warp_decoder.append(3)
        super(MobileUNetWarp3, self).__init__(target_size, n_classes, debug_samples=debug_samples)


class MobileUNetWarp2(MobileUNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0):
        self.warp_decoder.append(2)
        super(MobileUNetWarp2, self).__init__(target_size, n_classes, debug_samples=debug_samples)


class MobileUNetWarp1(MobileUNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0):
        self.warp_decoder.append(1)
        super(MobileUNetWarp1, self).__init__(target_size, n_classes, debug_samples=debug_samples)


class MobileUNetWarp24(MobileUNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0):
        self.warp_decoder.append(2)
        self.warp_decoder.append(4)
        super(MobileUNetWarp24, self).__init__(target_size, n_classes, debug_samples=debug_samples)


class MobileUNetWarp124(MobileUNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0):
        self.warp_decoder.append(1)
        self.warp_decoder.append(2)
        self.warp_decoder.append(4)
        super(MobileUNetWarp124, self).__init__(target_size, n_classes, debug_samples=debug_samples)


class MobileUNetWarp0(MobileUNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0):
        self.warp_decoder.append(0)
        super(MobileUNetWarp0, self).__init__(target_size, n_classes, debug_samples=debug_samples)


class MobileUNetWarpInp(MobileUNetWarp):
    def __init__(self, target_size, n_classes, debug_samples=0):
        self.warp_decoder.append(-1)
        super(MobileUNetWarpInp, self).__init__(target_size, n_classes, debug_samples=debug_samples)


if __name__ == '__main__':
    target_size = 256, 512
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = MobileUNetWarp0(target_size, 32)
    print(model.summary())
    model.plot_model()
