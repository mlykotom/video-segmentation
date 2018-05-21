from keras.layers import Activation, BatchNormalization
from keras.layers import Add
from keras.layers import AveragePooling2D
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.models import Model

from base_model import BaseModel
from layers import BilinearUpSampling2D, ResizeBilinear


class ICNet(BaseModel):
    def _prepare(self):
        self.input_shape = self.target_size + (3,)
        super(ICNet, self)._prepare()

    def branch_half(self, input_shape, prefix=''):
        x = Input(input_shape)
        y = ResizeBilinear(out_size=(int(x.shape[1]) // 2, int(x.shape[2]) // 2), name=prefix + 'data_sub2')(x)
        y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name=prefix + 'conv1_1_3x3_s2')(y)
        y = BatchNormalization(name=prefix + 'conv1_1_3x3_s2_bn')(y)
        y = Conv2D(32, 3, padding='same', activation='relu', name=prefix + 'conv1_2_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv1_2_3x3_s2_bn')(y)
        y = Conv2D(64, 3, padding='same', activation='relu', name=prefix + 'conv1_3_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv1_3_3x3_bn')(y)
        y_ = MaxPooling2D(pool_size=3, strides=2, name=prefix + 'pool1_3x3_s2')(y)

        y = Conv2D(128, 1, name=prefix + 'conv2_1_1x1_proj')(y_)
        y = BatchNormalization(name=prefix + 'conv2_1_1x1_proj_bn')(y)

        y_ = Conv2D(32, 1, activation='relu', name=prefix + 'conv2_1_1x1_reduce')(y_)
        y_ = BatchNormalization(name=prefix + 'conv2_1_1x1_reduce_bn')(y_)
        y_ = ZeroPadding2D(name=prefix + 'padding1')(y_)
        y_ = Conv2D(32, 3, activation='relu', name=prefix + 'conv2_1_3x3')(y_)
        y_ = BatchNormalization(name=prefix + 'conv2_1_3x3_bn')(y_)
        y_ = Conv2D(128, 1, name=prefix + 'conv2_1_1x1_increase')(y_)
        y_ = BatchNormalization(name=prefix + 'conv2_1_1x1_increase_bn')(y_)
        y = Add(name=prefix + 'conv2_1')([y, y_])
        y_ = Activation('relu', name=prefix + 'conv2_1/relu')(y)

        y = Conv2D(32, 1, activation='relu', name=prefix + 'conv2_2_1x1_reduce')(y_)
        y = BatchNormalization(name=prefix + 'conv2_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name=prefix + 'padding2')(y)
        y = Conv2D(32, 3, activation='relu', name=prefix + 'conv2_2_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv2_2_3x3_bn')(y)
        y = Conv2D(128, 1, name=prefix + 'conv2_2_1x1_increase')(y)
        y = BatchNormalization(name=prefix + 'conv2_2_1x1_increase_bn')(y)
        y = Add(name=prefix + 'conv2_2')([y, y_])
        y_ = Activation('relu', name=prefix + 'conv2_2/relu')(y)

        y = Conv2D(32, 1, activation='relu', name=prefix + 'conv2_3_1x1_reduce')(y_)
        y = BatchNormalization(name=prefix + 'conv2_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name=prefix + 'padding3')(y)
        y = Conv2D(32, 3, activation='relu', name=prefix + 'conv2_3_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv2_3_3x3_bn')(y)
        y = Conv2D(128, 1, name=prefix + 'conv2_3_1x1_increase')(y)
        y = BatchNormalization(name=prefix + 'conv2_3_1x1_increase_bn')(y)
        y = Add(name=prefix + 'conv2_3')([y, y_])
        y_ = Activation('relu', name=prefix + 'conv2_3/relu')(y)

        y = Conv2D(256, 1, strides=2, name=prefix + 'conv3_1_1x1_proj')(y_)
        y = BatchNormalization(name=prefix + 'conv3_1_1x1_proj_bn')(y)
        y_ = Conv2D(64, 1, strides=2, activation='relu', name=prefix + 'conv3_1_1x1_reduce')(y_)
        y_ = BatchNormalization(name=prefix + 'conv3_1_1x1_reduce_bn')(y_)
        y_ = ZeroPadding2D(name=prefix + 'padding4')(y_)
        y_ = Conv2D(64, 3, activation='relu', name=prefix + 'conv3_1_3x3')(y_)
        y_ = BatchNormalization(name=prefix + 'conv3_1_3x3_bn')(y_)
        y_ = Conv2D(256, 1, name=prefix + 'conv3_1_1x1_increase')(y_)
        y_ = BatchNormalization(name=prefix + 'conv3_1_1x1_increase_bn')(y_)
        y = Add(name=prefix + 'conv3_1')([y, y_])
        z = Activation('relu', name=prefix + 'conv3_1/relu')(y)

        return Model(x, z, name='branch_12')

    def branch_quarter(self, input_shape, prefix=''):
        z = Input(input_shape)
        y_ = ResizeBilinear(out_size=(int(z.shape[1]) // 2, int(z.shape[2]) // 2), name=prefix + 'conv3_1_sub4')(z)
        y = Conv2D(64, 1, activation='relu', name=prefix + 'conv3_2_1x1_reduce')(y_)
        y = BatchNormalization(name=prefix + 'conv3_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name=prefix + 'padding5')(y)
        y = Conv2D(64, 3, activation='relu', name=prefix + 'conv3_2_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv3_2_3x3_bn')(y)
        y = Conv2D(256, 1, name=prefix + 'conv3_2_1x1_increase')(y)
        y = BatchNormalization(name=prefix + 'conv3_2_1x1_increase_bn')(y)
        y = Add(name=prefix + 'conv3_2')([y, y_])
        y_ = Activation('relu', name=prefix + 'conv3_2/relu')(y)

        y = Conv2D(64, 1, activation='relu', name=prefix + 'conv3_3_1x1_reduce')(y_)
        y = BatchNormalization(name=prefix + 'conv3_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name=prefix + 'padding6')(y)
        y = Conv2D(64, 3, activation='relu', name=prefix + 'conv3_3_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv3_3_3x3_bn')(y)
        y = Conv2D(256, 1, name=prefix + 'conv3_3_1x1_increase')(y)
        y = BatchNormalization(name=prefix + 'conv3_3_1x1_increase_bn')(y)
        y = Add(name=prefix + 'conv3_3')([y, y_])
        y_ = Activation('relu', name=prefix + 'conv3_3/relu')(y)

        y = Conv2D(64, 1, activation='relu', name=prefix + 'conv3_4_1x1_reduce')(y_)
        y = BatchNormalization(name=prefix + 'conv3_4_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name=prefix + 'padding7')(y)
        y = Conv2D(64, 3, activation='relu', name=prefix + 'conv3_4_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv3_4_3x3_bn')(y)
        y = Conv2D(256, 1, name=prefix + 'conv3_4_1x1_increase')(y)
        y = BatchNormalization(name=prefix + 'conv3_4_1x1_increase_bn')(y)
        y = Add(name=prefix + 'conv3_4')([y, y_])
        y_ = Activation('relu', name=prefix + 'conv3_4/relu')(y)

        y = Conv2D(512, 1, name=prefix + 'conv4_1_1x1_proj')(y_)
        y = BatchNormalization(name=prefix + 'conv4_1_1x1_proj_bn')(y)
        y_ = Conv2D(128, 1, activation='relu', name=prefix + 'conv4_1_1x1_reduce')(y_)
        y_ = BatchNormalization(name=prefix + 'conv4_1_1x1_reduce_bn')(y_)
        y_ = ZeroPadding2D(padding=2, name=prefix + 'padding8')(y_)
        y_ = Conv2D(128, 3, dilation_rate=2, activation='relu', name=prefix + 'conv4_1_3x3')(y_)
        y_ = BatchNormalization(name=prefix + 'conv4_1_3x3_bn')(y_)
        y_ = Conv2D(512, 1, name=prefix + 'conv4_1_1x1_increase')(y_)
        y_ = BatchNormalization(name=prefix + 'conv4_1_1x1_increase_bn')(y_)
        y = Add(name=prefix + 'conv4_1')([y, y_])
        y_ = Activation('relu', name=prefix + 'conv4_1/relu')(y)

        y = Conv2D(128, 1, activation='relu', name=prefix + 'conv4_2_1x1_reduce')(y_)
        y = BatchNormalization(name=prefix + 'conv4_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name=prefix + 'padding9')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name=prefix + 'conv4_2_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv4_2_3x3_bn')(y)
        y = Conv2D(512, 1, name=prefix + 'conv4_2_1x1_increase')(y)
        y = BatchNormalization(name=prefix + 'conv4_2_1x1_increase_bn')(y)
        y = Add(name=prefix + 'conv4_2')([y, y_])
        y_ = Activation('relu', name=prefix + 'conv4_2/relu')(y)

        y = Conv2D(128, 1, activation='relu', name=prefix + 'conv4_3_1x1_reduce')(y_)
        y = BatchNormalization(name=prefix + 'conv4_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name=prefix + 'padding10')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name=prefix + 'conv4_3_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv4_3_3x3_bn')(y)
        y = Conv2D(512, 1, name=prefix + 'conv4_3_1x1_increase')(y)
        y = BatchNormalization(name=prefix + 'conv4_3_1x1_increase_bn')(y)
        y = Add(name=prefix + 'conv4_3')([y, y_])
        y_ = Activation('relu', name=prefix + 'conv4_3/relu')(y)

        y = Conv2D(128, 1, activation='relu', name=prefix + 'conv4_4_1x1_reduce')(y_)
        y = BatchNormalization(name=prefix + 'conv4_4_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name=prefix + 'padding11')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name=prefix + 'conv4_4_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv4_4_3x3_bn')(y)
        y = Conv2D(512, 1, name=prefix + 'conv4_4_1x1_increase')(y)
        y = BatchNormalization(name=prefix + 'conv4_4_1x1_increase_bn')(y)
        y = Add(name=prefix + 'conv4_4')([y, y_])
        y_ = Activation('relu', name=prefix + 'conv4_4/relu')(y)

        y = Conv2D(128, 1, activation='relu', name=prefix + 'conv4_5_1x1_reduce')(y_)
        y = BatchNormalization(name=prefix + 'conv4_5_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name=prefix + 'padding12')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name=prefix + 'conv4_5_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv4_5_3x3_bn')(y)
        y = Conv2D(512, 1, name=prefix + 'conv4_5_1x1_increase')(y)
        y = BatchNormalization(name=prefix + 'conv4_5_1x1_increase_bn')(y)
        y = Add(name=prefix + 'conv4_5')([y, y_])
        y_ = Activation('relu', name=prefix + 'conv4_5/relu')(y)

        y = Conv2D(128, 1, activation='relu', name=prefix + 'conv4_6_1x1_reduce')(y_)
        y = BatchNormalization(name=prefix + 'conv4_6_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name=prefix + 'padding13')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name=prefix + 'conv4_6_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv4_6_3x3_bn')(y)
        y = Conv2D(512, 1, name=prefix + 'conv4_6_1x1_increase')(y)
        y = BatchNormalization(name=prefix + 'conv4_6_1x1_increase_bn')(y)
        y = Add(name=prefix + 'conv4_6')([y, y_])
        y = Activation('relu', name=prefix + 'conv4_6/relu')(y)

        y_ = Conv2D(1024, 1, name=prefix + 'conv5_1_1x1_proj')(y)
        y_ = BatchNormalization(name=prefix + 'conv5_1_1x1_proj_bn')(y_)
        y = Conv2D(256, 1, activation='relu', name=prefix + 'conv5_1_1x1_reduce')(y)
        y = BatchNormalization(name=prefix + 'conv5_1_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=4, name=prefix + 'padding14')(y)
        y = Conv2D(256, 3, dilation_rate=4, activation='relu', name=prefix + 'conv5_1_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv5_1_3x3_bn')(y)
        y = Conv2D(1024, 1, name=prefix + 'conv5_1_1x1_increase')(y)
        y = BatchNormalization(name=prefix + 'conv5_1_1x1_increase_bn')(y)
        y = Add(name=prefix + 'conv5_1')([y, y_])
        y_ = Activation('relu', name=prefix + 'conv5_1/relu')(y)

        y = Conv2D(256, 1, activation='relu', name=prefix + 'conv5_2_1x1_reduce')(y_)
        y = BatchNormalization(name=prefix + 'conv5_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=4, name=prefix + 'padding15')(y)
        y = Conv2D(256, 3, dilation_rate=4, activation='relu', name=prefix + 'conv5_2_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv5_2_3x3_bn')(y)
        y = Conv2D(1024, 1, name=prefix + 'conv5_2_1x1_increase')(y)
        y = BatchNormalization(name=prefix + 'conv5_2_1x1_increase_bn')(y)
        y = Add(name=prefix + 'conv5_2')([y, y_])
        y_ = Activation('relu', name=prefix + 'conv5_2/relu')(y)

        y = Conv2D(256, 1, activation='relu', name=prefix + 'conv5_3_1x1_reduce')(y_)
        y = BatchNormalization(name=prefix + 'conv5_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=4, name=prefix + 'padding16')(y)
        y = Conv2D(256, 3, dilation_rate=4, activation='relu', name=prefix + 'conv5_3_3x3')(y)
        y = BatchNormalization(name=prefix + 'conv5_3_3x3_bn')(y)
        y = Conv2D(1024, 1, name=prefix + 'conv5_3_1x1_increase')(y)
        y = BatchNormalization(name=prefix + 'conv5_3_1x1_increase_bn')(y)
        y = Add(name=prefix + 'conv5_3')([y, y_])
        y = Activation('relu', name=prefix + 'conv5_3/relu')(y)

        return Model(z, y, name='branch_14')

    def pyramid_block(self, input_shape, prefix=''):
        input = Input(input_shape)
        h, w = input.shape[1:3].as_list()
        pool1 = AveragePooling2D(pool_size=(h, w), strides=(h, w), name=prefix + 'conv5_3_pool1')(input)
        pool1 = ResizeBilinear(out_size=(h, w), name=prefix + 'conv5_3_pool1_interp')(pool1)

        pool2 = AveragePooling2D(pool_size=(h / 2, w / 2), strides=(h // 2, w // 2), name=prefix + 'conv5_3_pool2')(input)
        pool2 = ResizeBilinear(out_size=(h, w), name=prefix + 'conv5_3_pool2_interp')(pool2)

        pool3 = AveragePooling2D(pool_size=(h / 3, w / 3), strides=(h // 3, w // 3), name=prefix + 'conv5_3_pool3')(input)
        pool3 = ResizeBilinear(out_size=(h, w), name=prefix + 'conv5_3_pool3_interp')(pool3)

        pool6 = AveragePooling2D(pool_size=(h / 4, w / 4), strides=(h // 4, w // 4), name=prefix + 'conv5_3_pool6')(input)
        pool6 = ResizeBilinear(out_size=(h, w), name=prefix + 'conv5_3_pool6_interp')(pool6)

        y = Add(name=prefix + 'conv5_3_sum')([input, pool1, pool2, pool3, pool6])
        y = Conv2D(256, 1, activation='relu', name=prefix + 'conv5_4_k1')(y)
        y = BatchNormalization(name=prefix + 'conv5_4_k1_bn')(y)

        aux_1 = BilinearUpSampling2D(name=prefix + 'conv5_4_interp')(y)
        return Model(input, aux_1, name='pyramid_block')

    def block_0(self, input_shape, prefix=''):
        x = Input(input_shape)
        y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name=prefix + 'conv1_sub1')(x)
        y = BatchNormalization(name=prefix + 'conv1_sub1_bn')(y)
        y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name=prefix + 'conv2_sub1')(y)
        y = BatchNormalization(name=prefix + 'conv2_sub1_bn')(y)
        y = Conv2D(64, 3, strides=2, padding='same', activation='relu', name=prefix + 'conv3_sub1')(y)
        y = BatchNormalization(name=prefix + 'conv3_sub1_bn')(y)
        y = Conv2D(128, 1, name=prefix + 'conv3_sub1_proj')(y)
        y = BatchNormalization(name=prefix + 'conv3_sub1_proj_bn')(y)
        return Model(x, y, name='branch_1')

    def out_block(self, y, aux_1, aux_2):
        if self.training_phase:
            out = Conv2D(self.n_classes, 1, activation='softmax', name='out')(y)  # conv6_cls
            aux_1 = Conv2D(self.n_classes, 1, activation='softmax', name='sub4_out')(aux_1)
            aux_2 = Conv2D(self.n_classes, 1, activation='softmax', name='sub24_out')(aux_2)

            return [out, aux_2, aux_1]
        else:
            out = Conv2D(self.n_classes, 1, activation='softmax', name='out')(y)  # conv6_cls
            out = BilinearUpSampling2D(size=(4, 4), name='out_full')(out)

            return [out]

    def _create_model(self):
        inp = Input(shape=self.target_size + (3,))
        x = inp

        # (1/2)
        branch_half = self.branch_half(self.input_shape)
        z = branch_half(x)

        # (1/4)
        branch_quarter = self.branch_quarter(branch_half.output_shape[1:])
        y = branch_quarter(z)

        pyramid_block = self.pyramid_block(branch_quarter.output_shape[1:])
        aux_1 = pyramid_block(y)

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
        block_0 = self.block_0(self.input_shape)
        y = block_0(x)

        y = Add(name='sub12_sum')([y, y_])
        y = Activation('relu', name='sub12_sum/relu')(y)
        y = BilinearUpSampling2D(name='sub12_sum_interp')(y)

        outputs = self.out_block(y, aux_1, aux_2)

        return Model(inputs=inp, outputs=outputs)

    @staticmethod
    def get_custom_objects():
        custom_objects = BaseModel.get_custom_objects()
        custom_objects.update({
            'BilinearUpSampling2D': BilinearUpSampling2D,
            'ResizeBilinear': ResizeBilinear,
        })
        return custom_objects

    def metrics(self):
        import metrics

        return {
            'out': [
                metrics.mean_iou,
            ]
        }

    def loss_weights(self):
        if self.training_phase:
            return [1.0, 0.4, 0.16]
        else:
            return None


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

    model = ICNet(target_size, 35, for_training=False)
    print(model.summary())
    model.plot_model()
