from keras import Input
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, K
from keras.models import Model

from base_model import BaseModel


class SegNet(BaseModel):
    def __init__(self, target_size, n_classes, debug_samples=0, for_training=True):
        self._filter_size = 64
        self._pool_size = (2, 2)
        self._kernel_size = (3, 3)
        self.input_shape = target_size + (3,)
        super(SegNet, self).__init__(target_size, n_classes, debug_samples, for_training)

    def _block(self, input, filter_size, kernel_size, pool_size):
        out = Convolution2D(filter_size, kernel_size, padding='same')(input)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        if pool_size is not None:
            out = MaxPooling2D(pool_size=pool_size)(out)
        return out

    def block_model(self, input_shape, filter_size, block_id, pool_at_end=True):
        # input_name = 'input_block_' + str(block_id)
        # name = input_name + '_' + str(K.get_uid(input_name))
        input = Input(shape=input_shape)
        out = Convolution2D(filter_size, self._kernel_size, padding='same')(input)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        if pool_at_end:
            out = MaxPooling2D(pool_size=self._pool_size)(out)
        return Model(input, out, name='conv_block_%d' % block_id)

    def _create_model(self):
        input = Input(shape=self.target_size + (3,), name='data_0')

        block_0 = self.block_model(self.input_shape, 64, 1, True)
        block_1 = self.block_model(block_0.output_shape[1:], 128, 2, True)
        block_2 = self.block_model(block_1.output_shape[1:], 256, 3, True)
        block_3 = self.block_model(block_2.output_shape[1:], 512, 4, False)

        out = block_0(input)
        out = block_1(out)
        out = block_2(out)
        out = block_3(out)

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

        model = Model(input, out)
        return model


if __name__ == '__main__':
    target_size = 256, 512
    model = SegNet(target_size, 32)

    print(model.summary())
    model.plot_model()
