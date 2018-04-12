import keras
from keras import Input
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Reshape, \
    SpatialDropout2D
from keras.models import Model

from base_model import BaseModel


class SegNet(BaseModel):

    def _create_model(self):
        # model = models.Sequential()
        # model.add(Layer(input_shape=(self.target_size[0], self.target_size[1], 3)))

        filter_size = 64
        pool_size = (2, 2)
        kernel_size = (3, 3)

        input = Input(shape=self.target_size + (3,), name='data_0')

        out = Convolution2D(filter_size, kernel_size, padding='same')(input)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = MaxPooling2D(pool_size=pool_size)(out)

        out = Convolution2D(128, kernel_size, padding='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = MaxPooling2D(pool_size=pool_size)(out)

        out = Convolution2D(256, kernel_size, padding='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = MaxPooling2D(pool_size=pool_size)(out)

        out = Convolution2D(512, kernel_size, padding='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = SpatialDropout2D(0.4)(out)

        # decoder
        out = Convolution2D(512, kernel_size, padding='same')(out)
        out = BatchNormalization()(out)

        out = UpSampling2D(size=pool_size)(out)
        out = Convolution2D(256, kernel_size, padding='same')(out)
        out = BatchNormalization()(out)

        out = UpSampling2D(size=pool_size)(out)
        out = Convolution2D(128, kernel_size, padding='same')(out)
        out = BatchNormalization()(out)

        out = UpSampling2D(size=pool_size)(out)
        out = Convolution2D(filter_size, kernel_size, padding='same')(out)
        out = BatchNormalization()(out)

        out = Convolution2D(self.n_classes, (1, 1), padding='same')(out)

        out = Reshape((-1, self.n_classes))(out)
        out = Activation('softmax')(out)

        model = Model(input, out)

        return model


if __name__ == '__main__':
    target_size = (288, 480)
    model = SegNet(target_size, 30)

    print(model.summary())
    keras.utils.plot_model(model.k, 'segnet.png', show_shapes=True, show_layer_names=True)
