import os

from keras.applications import NASNetMobile

from models import BaseModel

# TODO urcite nope nope nope nope nope .. model je sileny :D

class NasNet(BaseModel):
    def _create_model(self):
        model = NASNetMobile(
            input_shape=(self.target_size[0], self.target_size[1], 3),
            include_top=False,
        )

        return model