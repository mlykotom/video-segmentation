import keras
import tensorflow as tf
from keras.layers import Activation, BatchNormalization
from keras.layers import Add
from keras.layers import AveragePooling2D
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.models import Model

from base_model import BaseModel
from layers import BilinearUpSampling2D


class ICNet(BaseModel):

    def branch_half(self, x):
        y = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) // 2, int(x.shape[2]) // 2)), name='data_sub2')(x)
        y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv1_1_3x3_s2')(y)
        y = BatchNormalization(name='conv1_1_3x3_s2_bn')(y)
        y = Conv2D(32, 3, padding='same', activation='relu', name='conv1_2_3x3')(y)
        y = BatchNormalization(name='conv1_2_3x3_s2_bn')(y)
        y = Conv2D(64, 3, padding='same', activation='relu', name='conv1_3_3x3')(y)
        y = BatchNormalization(name='conv1_3_3x3_bn')(y)
        y_ = MaxPooling2D(pool_size=3, strides=2, name='pool1_3x3_s2')(y)

        y = Conv2D(128, 1, name='conv2_1_1x1_proj')(y_)
        y = BatchNormalization(name='conv2_1_1x1_proj_bn')(y)

        y_ = Conv2D(32, 1, activation='relu', name='conv2_1_1x1_reduce')(y_)
        y_ = BatchNormalization(name='conv2_1_1x1_reduce_bn')(y_)
        y_ = ZeroPadding2D(name='padding1')(y_)
        y_ = Conv2D(32, 3, activation='relu', name='conv2_1_3x3')(y_)
        y_ = BatchNormalization(name='conv2_1_3x3_bn')(y_)
        y_ = Conv2D(128, 1, name='conv2_1_1x1_increase')(y_)
        y_ = BatchNormalization(name='conv2_1_1x1_increase_bn')(y_)
        y = Add(name='conv2_1')([y, y_])
        y_ = Activation('relu', name='conv2_1/relu')(y)

        y = Conv2D(32, 1, activation='relu', name='conv2_2_1x1_reduce')(y_)
        y = BatchNormalization(name='conv2_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding2')(y)
        y = Conv2D(32, 3, activation='relu', name='conv2_2_3x3')(y)
        y = BatchNormalization(name='conv2_2_3x3_bn')(y)
        y = Conv2D(128, 1, name='conv2_2_1x1_increase')(y)
        y = BatchNormalization(name='conv2_2_1x1_increase_bn')(y)
        y = Add(name='conv2_2')([y, y_])
        y_ = Activation('relu', name='conv2_2/relu')(y)

        y = Conv2D(32, 1, activation='relu', name='conv2_3_1x1_reduce')(y_)
        y = BatchNormalization(name='conv2_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding3')(y)
        y = Conv2D(32, 3, activation='relu', name='conv2_3_3x3')(y)
        y = BatchNormalization(name='conv2_3_3x3_bn')(y)
        y = Conv2D(128, 1, name='conv2_3_1x1_increase')(y)
        y = BatchNormalization(name='conv2_3_1x1_increase_bn')(y)
        y = Add(name='conv2_3')([y, y_])
        y_ = Activation('relu', name='conv2_3/relu')(y)

        y = Conv2D(256, 1, strides=2, name='conv3_1_1x1_proj')(y_)
        y = BatchNormalization(name='conv3_1_1x1_proj_bn')(y)
        y_ = Conv2D(64, 1, strides=2, activation='relu', name='conv3_1_1x1_reduce')(y_)
        y_ = BatchNormalization(name='conv3_1_1x1_reduce_bn')(y_)
        y_ = ZeroPadding2D(name='padding4')(y_)
        y_ = Conv2D(64, 3, activation='relu', name='conv3_1_3x3')(y_)
        y_ = BatchNormalization(name='conv3_1_3x3_bn')(y_)
        y_ = Conv2D(256, 1, name='conv3_1_1x1_increase')(y_)
        y_ = BatchNormalization(name='conv3_1_1x1_increase_bn')(y_)
        y = Add(name='conv3_1')([y, y_])
        z = Activation('relu', name='conv3_1/relu')(y)

        return z

    def branch_quarter(self, z):
        y_ = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) // 2, int(x.shape[2]) // 2)), name='conv3_1_sub4')(z)
        y = Conv2D(64, 1, activation='relu', name='conv3_2_1x1_reduce')(y_)
        y = BatchNormalization(name='conv3_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding5')(y)
        y = Conv2D(64, 3, activation='relu', name='conv3_2_3x3')(y)
        y = BatchNormalization(name='conv3_2_3x3_bn')(y)
        y = Conv2D(256, 1, name='conv3_2_1x1_increase')(y)
        y = BatchNormalization(name='conv3_2_1x1_increase_bn')(y)
        y = Add(name='conv3_2')([y, y_])
        y_ = Activation('relu', name='conv3_2/relu')(y)

        y = Conv2D(64, 1, activation='relu', name='conv3_3_1x1_reduce')(y_)
        y = BatchNormalization(name='conv3_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding6')(y)
        y = Conv2D(64, 3, activation='relu', name='conv3_3_3x3')(y)
        y = BatchNormalization(name='conv3_3_3x3_bn')(y)
        y = Conv2D(256, 1, name='conv3_3_1x1_increase')(y)
        y = BatchNormalization(name='conv3_3_1x1_increase_bn')(y)
        y = Add(name='conv3_3')([y, y_])
        y_ = Activation('relu', name='conv3_3/relu')(y)

        y = Conv2D(64, 1, activation='relu', name='conv3_4_1x1_reduce')(y_)
        y = BatchNormalization(name='conv3_4_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding7')(y)
        y = Conv2D(64, 3, activation='relu', name='conv3_4_3x3')(y)
        y = BatchNormalization(name='conv3_4_3x3_bn')(y)
        y = Conv2D(256, 1, name='conv3_4_1x1_increase')(y)
        y = BatchNormalization(name='conv3_4_1x1_increase_bn')(y)
        y = Add(name='conv3_4')([y, y_])
        y_ = Activation('relu', name='conv3_4/relu')(y)

        y = Conv2D(512, 1, name='conv4_1_1x1_proj')(y_)
        y = BatchNormalization(name='conv4_1_1x1_proj_bn')(y)
        y_ = Conv2D(128, 1, activation='relu', name='conv4_1_1x1_reduce')(y_)
        y_ = BatchNormalization(name='conv4_1_1x1_reduce_bn')(y_)
        y_ = ZeroPadding2D(padding=2, name='padding8')(y_)
        y_ = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_1_3x3')(y_)
        y_ = BatchNormalization(name='conv4_1_3x3_bn')(y_)
        y_ = Conv2D(512, 1, name='conv4_1_1x1_increase')(y_)
        y_ = BatchNormalization(name='conv4_1_1x1_increase_bn')(y_)
        y = Add(name='conv4_1')([y, y_])
        y_ = Activation('relu', name='conv4_1/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_2_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding9')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_2_3x3')(y)
        y = BatchNormalization(name='conv4_2_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_2_1x1_increase')(y)
        y = BatchNormalization(name='conv4_2_1x1_increase_bn')(y)
        y = Add(name='conv4_2')([y, y_])
        y_ = Activation('relu', name='conv4_2/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_3_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding10')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_3_3x3')(y)
        y = BatchNormalization(name='conv4_3_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_3_1x1_increase')(y)
        y = BatchNormalization(name='conv4_3_1x1_increase_bn')(y)
        y = Add(name='conv4_3')([y, y_])
        y_ = Activation('relu', name='conv4_3/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_4_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_4_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding11')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_4_3x3')(y)
        y = BatchNormalization(name='conv4_4_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_4_1x1_increase')(y)
        y = BatchNormalization(name='conv4_4_1x1_increase_bn')(y)
        y = Add(name='conv4_4')([y, y_])
        y_ = Activation('relu', name='conv4_4/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_5_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_5_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding12')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_5_3x3')(y)
        y = BatchNormalization(name='conv4_5_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_5_1x1_increase')(y)
        y = BatchNormalization(name='conv4_5_1x1_increase_bn')(y)
        y = Add(name='conv4_5')([y, y_])
        y_ = Activation('relu', name='conv4_5/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_6_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_6_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding13')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_6_3x3')(y)
        y = BatchNormalization(name='conv4_6_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_6_1x1_increase')(y)
        y = BatchNormalization(name='conv4_6_1x1_increase_bn')(y)
        y = Add(name='conv4_6')([y, y_])
        y = Activation('relu', name='conv4_6/relu')(y)

        y_ = Conv2D(1024, 1, name='conv5_1_1x1_proj')(y)
        y_ = BatchNormalization(name='conv5_1_1x1_proj_bn')(y_)
        y = Conv2D(256, 1, activation='relu', name='conv5_1_1x1_reduce')(y)
        y = BatchNormalization(name='conv5_1_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=4, name='padding14')(y)
        y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_1_3x3')(y)
        y = BatchNormalization(name='conv5_1_3x3_bn')(y)
        y = Conv2D(1024, 1, name='conv5_1_1x1_increase')(y)
        y = BatchNormalization(name='conv5_1_1x1_increase_bn')(y)
        y = Add(name='conv5_1')([y, y_])
        y_ = Activation('relu', name='conv5_1/relu')(y)

        y = Conv2D(256, 1, activation='relu', name='conv5_2_1x1_reduce')(y_)
        y = BatchNormalization(name='conv5_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=4, name='padding15')(y)
        y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_2_3x3')(y)
        y = BatchNormalization(name='conv5_2_3x3_bn')(y)
        y = Conv2D(1024, 1, name='conv5_2_1x1_increase')(y)
        y = BatchNormalization(name='conv5_2_1x1_increase_bn')(y)
        y = Add(name='conv5_2')([y, y_])
        y_ = Activation('relu', name='conv5_2/relu')(y)

        y = Conv2D(256, 1, activation='relu', name='conv5_3_1x1_reduce')(y_)
        y = BatchNormalization(name='conv5_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=4, name='padding16')(y)
        y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_3_3x3')(y)
        y = BatchNormalization(name='conv5_3_3x3_bn')(y)
        y = Conv2D(1024, 1, name='conv5_3_1x1_increase')(y)
        y = BatchNormalization(name='conv5_3_1x1_increase_bn')(y)
        y = Add(name='conv5_3')([y, y_])
        y = Activation('relu', name='conv5_3/relu')(y)
        return y

    def pyramid_block(self, y):
        h, w = y.shape[1:3].as_list()
        pool1 = AveragePooling2D(pool_size=(h, w), strides=(h, w), name='conv5_3_pool1')(y)
        pool1 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool1_interp')(pool1)
        pool2 = AveragePooling2D(pool_size=(h / 2, w / 2), strides=(h // 2, w // 2), name='conv5_3_pool2')(y)
        pool2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool2_interp')(pool2)
        pool3 = AveragePooling2D(pool_size=(h / 3, w / 3), strides=(h // 3, w // 3), name='conv5_3_pool3')(y)
        pool3 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool3_interp')(pool3)
        pool6 = AveragePooling2D(pool_size=(h / 4, w / 4), strides=(h // 4, w // 4), name='conv5_3_pool6')(y)
        pool6 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool6_interp')(pool6)

        y = Add(name='conv5_3_sum')([y, pool1, pool2, pool3, pool6])
        y = Conv2D(256, 1, activation='relu', name='conv5_4_k1')(y)
        y = BatchNormalization(name='conv5_4_k1_bn')(y)

        aux_1 = BilinearUpSampling2D(name='conv5_4_interp')(y)
        return aux_1

    def block_0(self, x, prefix=''):
        y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name=prefix + 'conv1_sub1')(x)
        y = BatchNormalization(name=prefix + 'conv1_sub1_bn')(y)
        y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name=prefix + 'conv2_sub1')(y)
        y = BatchNormalization(name=prefix + 'conv2_sub1_bn')(y)
        y = Conv2D(64, 3, strides=2, padding='same', activation='relu', name=prefix + 'conv3_sub1')(y)
        y = BatchNormalization(name=prefix + 'conv3_sub1_bn')(y)
        y = Conv2D(128, 1, name=prefix + 'conv3_sub1_proj')(y)
        y = BatchNormalization(name=prefix + 'conv3_sub1_proj_bn')(y)
        return y

    def out_block(self, inputs, y, aux_1, aux_2):
        if self.training_phase:
            out = Conv2D(self.n_classes, 1, activation='softmax', name='out')(y)  # conv6_cls
            # out = Conv2D(self.n_classes, 1, padding='same', name='conv6_cls')(y)
            # out = Reshape((-1, self.n_classes))(out)
            # out = Activation('softmax', name='out')(out)

            aux_1 = Conv2D(self.n_classes, 1, activation='softmax', name='sub4_out')(aux_1)
            # aux_1 = Conv2D(self.n_classes, 1, padding='same', name='sub4_out')(aux_1)
            # aux_1 = Reshape((-1, self.n_classes))(aux_1)
            # aux_1 = Activation('softmax', name='out_aux_1')(aux_1)

            aux_2 = Conv2D(self.n_classes, 1, activation='softmax', name='sub24_out')(aux_2)
            # aux_2 = Conv2D(self.n_classes, 1, padding='same', name='sub24_out')(aux_2)
            # aux_2 = Reshape((-1, self.n_classes))(aux_2)
            # aux_2 = Activation('softmax', name='out_aux_2')(aux_2)

            model = Model(inputs=inputs, outputs=[out, aux_2, aux_1])
        else:
            out = Conv2D(self.n_classes, 1, activation='softmax', name='out')(y)  # conv6_cls
            out = BilinearUpSampling2D(size=(4, 4), name='out_full')(out)

            model = Model(inputs=inputs, outputs=out)

        return model

    def _create_model(self):
        inp = Input(shape=self.target_size + (3,))
        x = inp

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

        y = Add(name='sub12_sum')([y, y_])
        y = Activation('relu', name='sub12_sum/relu')(y)
        y = BilinearUpSampling2D(name='sub12_sum_interp')(y)

        return self.out_block(inp, y, aux_1, aux_2)

    def get_custom_objects(self):
        parent_objects = super(ICNet, self).get_custom_objects()
        parent_objects.update({
            'BilinearUpSampling2D': BilinearUpSampling2D,
        })
        return parent_objects

    def metrics(self):
        import metrics

        return {
            'out': [
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                keras.metrics.categorical_accuracy,
                metrics.mean_iou,
            ]
        }

    def compile(self):
        print("-- Optimizer: " + type(self.optimizer()).__name__)

        if self.training_phase:
            phase_loss_weights = [1.0, 0.4, 0.16]
        else:
            phase_loss_weights = None

        self._model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=self.optimizer(),
            metrics=self.metrics(),
            loss_weights=phase_loss_weights
        )

    def optimizer_params(self):
        # return optimizers.Adam(lr=0.0001) # baseline_b6_lr0.0001
        # return optimizers.Adam(decay=0.00006)
        # return optimizers.Adam()
        # return optimizers.Adam(lr=0.0001, decay=0.00001)
        # return optimizers.Adam(lr=0.0001, decay=0.0008)
        # return optimizers.Adam(lr=0.0001, decay=0.001)
        # return optimizers.Adam(lr=0.00025, decay=0.0099)
        # return optimizers.Adam(lr=0.0001, decay=0.003) # TODO: 2nd best
        # return optimizers.Adam(lr=0.0002, decay=0.099)  # TODO this is the best

        if self.is_debug:
            # return {'lr': 0.0002, 'decay': 0.0991} # for 120 samples
            # return {'lr': 0.000305, 'decay': 0.0991} # for 20 samples
            return {'lr': 0.00031, 'decay': 0.0999}  # for 20 samples
        else:
            # return {'lr': 0.0009, 'decay': 0.005}
            # return {'lr': 0.001, 'decay': 0.005} # running on mlyko.can
            # return {'lr': 0.001, 'decay': 0.01}  # running on mlyko.can
            # return {'lr': 0.001, 'decay': 0.009}  # running on mlyko.can
            # return {'lr': 0.0018, 'decay': 0.001}  # running on doom
            # return {'lr': 0.001, 'decay': 0.009} # runs on doom
            # return {'lr': 0.0017, 'decay': 0.001}  # running on mlyko.can
            return {'lr': 0.002, 'decay': 0.002}  # running on mlyko.can


if __name__ == '__main__':
    target_size = 256, 512
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = ICNet(target_size, 32, for_training=False)
    print(model.summary())
    model.plot_model()
