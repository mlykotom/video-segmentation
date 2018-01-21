from time import gmtime, strftime

from keras.callbacks import ModelCheckpoint

import cityscapes_labels
import data_generator
from callbacks import *
from model import segnet


def train(images_path, labels_path, run_name):
    # target_height, target_width = 239, 253
    # target_height, target_width = 360, 480
    target_height, target_width = 360, 648

    labels = cityscapes_labels.labels
    n_classes = len(labels)

    batch_size = 2
    epochs = 100

    model = segnet.get_model(target_height, target_width, n_classes)
    model.summary()

    model.compile(
        loss="categorical_crossentropy",
        optimizer='adam',
        metrics=["categorical_accuracy"]
    )

    used_callbacks = []

    # ------------- lr scheduler
    lr_base = 0.01 * (float(batch_size) / 16)
    lr_power = 0.9
    used_callbacks.append(lr_scheduler(epochs, lr_base, lr_power))

    # ------------- tensorboard
    tb = tensorboard('../logs', run_name, histogram_freq=0)
    used_callbacks.append(tb)

    # ------------- model checkpoint
    filepath = "weights/" + run_name + "_{epoch:02d}-{categorical_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='categorical_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    used_callbacks.append(checkpoint)

    # ------------- data generator
    datagen = data_generator.SimpleSegmentationGenerator(
        images_path=images_path,
        labels_path=labels_path,
        validation_split=0.2,
        debug_samples=20
    )

    # ------------- fit!
    model.fit_generator(
        generator=datagen.training_flow(labels, batch_size, (target_height, target_width)),
        steps_per_epoch=datagen.steps_per_epoch(batch_size),
        epochs=epochs,
        verbose=1,
        validation_data=datagen.validation_flow(labels, batch_size, (target_height, target_width)),
        validation_steps=datagen.validation_steps(batch_size),
        callbacks=used_callbacks
    )

    # save final model
    model.save_weights('weights/' + run_name + '_' + str(epochs) + '_finished.h5')


if __name__ == '__main__':
    run_started = 'gta_segnet_' + strftime("%Y_%m_%d_%H:%M", gmtime()) + '_'

    try:
        # tries to get dataset path from os environments
        dataset_path = os.environ['DATASETS']
    except:
        dataset_path = '/home/xmlyna06/data/gta/'

    images_path = os.path.join(dataset_path, 'images/')
    labels_path = os.path.join(dataset_path, 'labels/')

    train(images_path, labels_path, run_started)
