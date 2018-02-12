from keras import models
from keras.engine import Layer
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Reshape

from base_model import BaseModel


class SegNet(BaseModel):

    def _create_model(self):
        model = models.Sequential()
        model.add(Layer(input_shape=(self.target_size[0], self.target_size[1], 3)))

        filter_size = 64
        pool_size = (2, 2)
        kernel_size = (3, 3)

        # encoder
        model.add(Convolution2D(filter_size, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))

        model.add(Convolution2D(128, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))

        model.add(Convolution2D(256, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))

        model.add(Convolution2D(512, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # decoder
        model.add(Convolution2D(512, kernel_size, padding='same'))
        model.add(BatchNormalization())

        model.add(UpSampling2D(size=pool_size))
        model.add(Convolution2D(256, kernel_size, padding='same'))
        model.add(BatchNormalization())

        model.add(UpSampling2D(size=pool_size))
        model.add(Convolution2D(128, kernel_size, padding='same'))
        model.add(BatchNormalization())

        model.add(UpSampling2D(size=pool_size))
        model.add(Convolution2D(filter_size, kernel_size, padding='same'))
        model.add(BatchNormalization())

        model.add(Convolution2D(self.n_classes, (1, 1), padding='same'))

        model.add(Reshape((-1, self.n_classes)))
        model.add(Activation('softmax'))

        return model


if __name__ == '__main__':
    model = SegNet((360, 480), 30)

    model.summary()
