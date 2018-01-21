from time import gmtime, strftime

from keras.callbacks import ModelCheckpoint

from callbacks import *
from model import segnet
import cityscapes_labels, data_generator

run_started = 'gta_segnet_' + strftime("%Y-%m-%d-%H:%M", gmtime())

if __name__ == '__main__':
    # target_height, target_width = 239, 253
    # target_height, target_width = 360, 480
    target_height, target_width = 360, 648

    labels = cityscapes_labels.labels
    n_classes = len(labels)

    batch_size = 2
    epochs = 100

    dataset_path = '/home/xmlyna06/data/gta/'

    images_path = os.path.join(dataset_path, 'images/')
    labels_path = os.path.join(dataset_path, 'labels/')

    model = segnet.get_model(target_height, target_width, n_classes)
    model.summary()

    model.compile(
        loss="categorical_crossentropy",
        optimizer='adam',
        metrics=["categorical_accuracy"]
    )

    print(model.summary())

    used_callbacks = []

    # ------------- lr scheduler
    lr_base = 0.01 * (float(batch_size) / 16)
    lr_power = 0.9
    used_callbacks.append(lr_scheduler(epochs, lr_base, lr_power))

    # ------------- tensorboard
    tb = tensorboard('../logs', run_started, histogram_freq=0)
    used_callbacks.append(tb)

    # ------------- model checkpoint
    run_started = ''
    filepath = "weights/weights_" + run_started + "_{epoch:02d}-{categorical_accuracy:.2f}.hdf5"
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

    model.fit_generator(
        generator=datagen.training_flow(labels, batch_size, (target_height, target_width)),
        steps_per_epoch=datagen.steps_per_epoch(batch_size),
        epochs=epochs,
        verbose=1,
        validation_data=datagen.validation_flow(labels, batch_size, (target_height, target_width)),
        validation_steps=datagen.validation_steps(batch_size),
        callbacks=used_callbacks
    )

    model.save_weights('test_gta_weights.h5')
