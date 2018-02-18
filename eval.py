# model
import os

import keras
from keras import losses, optimizers

import cityscapes_labels
import config
import metrics
from data_generator import SimpleSegmentationGenerator
from models import MobileUNet

batch_size = 1
target_size = 384, 640
# target_size = (1052, 1914)
labels = cityscapes_labels.labels
n_classes = len(labels)
dataset_path = config.data_path('gta')
images_path = os.path.join(dataset_path, 'images/')
labels_path = os.path.join(dataset_path, 'labels/')

model = MobileUNet(target_size, n_classes)

model.k.load_weights('weights/MobileUNet_2018_02_15_09:34_cat_acc-0.89.hdf5')

# model.summary()

model.k.compile(
    loss=losses.categorical_crossentropy,
    optimizer=optimizers.Adam(lr=0.001),
    metrics=[
        metrics.dice_coef,
        metrics.precision,
        keras.metrics.categorical_accuracy
    ]
)

datagen = SimpleSegmentationGenerator(
    images_path=images_path,
    labels_path=labels_path
)


eval_batch_size = 4

prediction = model.k.evaluate_generator(
    generator=datagen.flow('test', labels, eval_batch_size, target_size),
    steps=datagen.steps_per_epoch('test', eval_batch_size),
)

print(model.k.metrics_names)
print(prediction)

model.k.predict()
