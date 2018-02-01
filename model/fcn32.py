from keras.engine import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Reshape, Activation

from gta_segnet.layers.BilinearUpSampling import BilinearUpSampling2D


def vgg_block(x, filter_size, block_n, conv_n):
    pool_size = (2, 2)
    kernel_size = (3, 3)

    block_name = 'block%d' % block_n

    for i in range(1, conv_n + 1):
        conv_name = '_conv%d' % i
        x = Conv2D(filter_size, kernel_size, activation='relu', padding='same', name=block_name + conv_name)(x)

    x = MaxPooling2D(pool_size, strides=(2, 2), name=block_name + '_pool')(x)
    return x


def get_model(target_height, target_width, n_classes):
    pass

    img_input = Input(shape=(target_height, target_width, 3))

    filter_size = 64
    x = vgg_block(img_input, filter_size, block_n=1, conv_n=2)

    filter_size *= 2
    x = vgg_block(x, filter_size, block_n=2, conv_n=2)

    filter_size *= 2
    x = vgg_block(x, filter_size, block_n=3, conv_n=3)

    filter_size *= 2
    x = vgg_block(x, filter_size, block_n=4, conv_n=3)

    # filter_size *= 2
    x = vgg_block(x, filter_size, block_n=5, conv_n=3)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1',
               # kernel_regularizer=l2(weight_decay)
               )(x)
    # x = Dropout(0.5)(x)

    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2',
               # kernel_regularizer=l2(weight_decay)
               )(x)
    # x = Dropout(0.5)(x)
    # classifying layer
    x = Conv2D(n_classes, (1, 1),
               kernel_initializer='he_normal',
               activation='linear',
               padding='valid',
               strides=(1, 1),
               # kernel_regularizer=l2(weight_decay)
               )(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)

    x = Reshape((-1, n_classes))(x)
    # x = Activation('softmax')(x)

    model = Model(img_input, x)
    return model


if __name__ == '__main__':
    model = get_model(352, 480, 12)

    model.summary()
