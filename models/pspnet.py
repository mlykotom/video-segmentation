from keras_contrib.applications import ResNet, ResNet34

from models import BaseModel


class PSPNet101(BaseModel):
    def _create_model(self):
        pass
        # input = Input(shape=(self.target_size[0], self.target_size[1], 3))

        # res = ResNet(input_shape, self.n_classes, bottleneck, repetitions=[3, 4, 23, 3])

        # res = ResNet101((224, 224, 3), self.n_classes)


if __name__ == '__main__':
    model = ResNet34(None, 1000)


    # resnet101 = ResNet(input_shape=(224, 224, 3), classes=10, repetitions=[3, 4, 23, 3], include_top=False)

