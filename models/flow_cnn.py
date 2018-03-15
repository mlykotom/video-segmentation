import keras
from keras import Input, Model
from keras.layers import Conv2D, concatenate, Reshape, Activation

from base_model import BaseModel


class FlowCNN(BaseModel):
    def _create_model(self):
        img_flo = Input(shape=(self.target_size[0], self.target_size[1], 2), name='flo')
        img_old = Input(shape=(self.target_size[0], self.target_size[1], 3), name='data_0')
        img_new = Input(shape=(self.target_size[0], self.target_size[1], 3), name='data_1')
        img_diff = Input(shape=(self.target_size[0], self.target_size[1], 3), name='data_diff')

        input = concatenate([img_flo, img_old, img_new, img_diff])

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)

        x = concatenate([img_flo, x])
        x = Conv2D(2, (3, 3), padding='same')(x)

        return Model([img_flo, img_old, img_new, img_diff], x)


if __name__ == '__main__':
    target_size = (288, 480)
    model = FlowCNN(target_size, 32)

    # print(model.summary())
    keras.utils.plot_model(model.k, 'flow_cnn.png', show_shapes=True, show_layer_names=True)