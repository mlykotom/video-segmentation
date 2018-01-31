from keras import Input
from keras.applications import VGG16
from keras.engine import Model
from keras.layers import Conv2D, Cropping2D, Add, Conv2DTranspose, Reshape


# crop o1 wrt o2
def crop(o1, o2, i):
    o_shape2 = Model(i, o2).output_shape
    outputHeight2 = o_shape2[1]
    outputWidth2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape
    outputHeight1 = o_shape1[1]
    outputWidth1 = o_shape1[2]

    print(o_shape1, o_shape2)

    cx = abs(outputWidth1 - outputWidth2)
    cy = abs(outputHeight2 - outputHeight1)

    if outputWidth1 > outputWidth2:
        o1 = Cropping2D(cropping=((0, 0), (0, cx)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0), (0, cx)))(o2)

    if outputHeight1 > outputHeight2:
        o1 = Cropping2D(cropping=((0, cy), (0, 0)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy), (0, 0)))(o2)

    return o1, o2


def connect_skip(model, img_input, layer_from, n_classes, name):
    x2 = layer_from
    x2 = (Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', name='score_' + name))(x2)
    model, x2 = crop(model, x2, img_input)
    model = Add(name=name)([x2, model])
    return model


def get_model(input_height, input_width, n_classes):
    img_input = Input(shape=(input_height, input_width, 3))
    vgg = VGG16(weights='imagenet', include_top=False, input_tensor=img_input)

    f1 = vgg.get_layer('block1_pool').output
    f2 = vgg.get_layer('block2_pool').output
    f3 = vgg.get_layer('block3_pool').output
    f4 = vgg.get_layer('block4_pool').output
    f5 = vgg.get_layer('block5_pool').output

    # fully connected (as convolution)
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1')(vgg.output)
    # x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(x)
    # x = Dropout(0.5)(x)
    x = Conv2D(n_classes, (1, 1), activation='linear', name='predictions_' + str(n_classes))(x)

    x = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(x)
    x = connect_skip(x, img_input, f4, n_classes, name='skip1')

    x = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(x)
    x = connect_skip(x, img_input, f3, n_classes, name='skip2')

    x = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False)(x)

    # TODO properly calculate how to crop
    # example here https://github.com/mzaradzki/neuralnets/blob/master/vgg_segmentation_keras/utils.py
    x = Cropping2D(cropping=((4, 4), (4, 4)))(x)
    x = Reshape((-1, n_classes))(x)

    model = Model(vgg.input, x)

    return model


if __name__ == '__main__':
    # test
    m = get_model(360, 648, 12)
    m.summary()
