from keras import models
from keras.engine import Layer
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Reshape


def get_model(input_height, input_width, n_classes):
    """
    :param input_height:
    :param input_width:
    :param n_classes:
    :return:
    """
    model = models.Sequential()
    model.add(Layer(input_shape=(input_height, input_width, 3)))

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

    model.add(Convolution2D(n_classes, (1, 1), padding='same'))

    model.add(Reshape((-1, n_classes)))
    model.add(Activation('softmax'))

    return model


if __name__ == '__main__':
    # test
    m = get_model(352, 480, 12)
    m.summary()
