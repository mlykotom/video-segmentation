import keras
from keras import Input, Model
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Reshape, \
    concatenate, Lambda

from base_model import BaseModel
from layers import tf_warp


class SegNetWarp(BaseModel):

    def first_block(self, input, filter_size, kernel_size, pool_size):
        out = Convolution2D(filter_size, kernel_size, padding='same')(input)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = MaxPooling2D(pool_size=pool_size)(out)
        return out

    def _create_model(self):
        filter_size = 64
        pool_size = (2, 2)
        kernel_size = (3, 3)

        img_old = Input(shape=self.target_size + (3,), name='data_0')
        img_new = Input(shape=self.target_size + (3,), name='data_1')
        flo = Input(shape=self.target_size + (2,), name='flow')

        all_inputs = [img_old, img_new, flo]

        def warp_test(x):
            img = x[0]
            flow = x[1]
            # TODO resize flow based on img shape

            out_size = img.get_shape().as_list()[1:3]
            out = tf_warp(img, flow, out_size)
            return out

        warped = Lambda(warp_test, name="warp")([img_old, flo])
        left_branch = self.first_block(warped, filter_size, kernel_size, pool_size)
        # left_branch = warped

        right_branch = self.first_block(img_new, filter_size, kernel_size, pool_size)

        # encoder
        out = concatenate([left_branch, right_branch])

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

        model = Model(inputs=all_inputs, outputs=[out])

        return model


if __name__ == '__main__':
    target_size = (288, 480)
    model = SegNetWarp(target_size, 32)

    print(model.summary())
    keras.utils.plot_model(model.k, 'segnet_warp.png', show_shapes=True, show_layer_names=True)
