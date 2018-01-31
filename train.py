from time import gmtime, strftime

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

import cityscapes_labels
import data_generator
from callbacks import *
from loss import precision, dice_coef
from model import segnet  # , fcn8, fcn32, mobile_unet
import keras.models

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

implemented_models = ['segnet', 'mobile_unet']


def train(images_path, labels_path, model_name='segnet', run_name=''):
    # target_height, target_width = 239, 253
    # target_height, target_width = 360, 480
    # target_height, target_width = 200, 224
    target_height, target_width = 360, 648
    # target_height, target_width = 352, 480

    labels = cityscapes_labels.labels
    n_classes = len(labels)

    batch_size = 2
    epochs = 200

    if model_name == 'segnet':
        model = segnet.get_model(target_height, target_width, n_classes)
    # elif model_name == 'fcn8':
    #     model = fcn8.get_model(target_height, target_width, n_classes)
    # elif model_name == 'fcn32':
    #     model = fcn32.get_model(target_height, target_width, n_classes)
    # elif model_name == 'mobile_unet':
    #     model = mobile_unet.get_model(target_height, target_width, n_classes)
    else:
        raise NotImplemented(
            'Model must be one of [' + ','.join(implemented_models) + ']')

    model.summary()
    plot_model(
        model,
        to_file='model_' + model_name + '.png',
        show_layer_names=True,
        show_shapes=True
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(lr=0.001),
        metrics=[
            dice_coef,
            precision,
            "categorical_accuracy"
        ]
    )

    # model.compile(
    #     optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
    #     # optimizer=Adam(lr=0.001),
    #     # optimizer=optimizers.RMSprop(),
    #     loss=dice_coef_loss,
    #     metrics=[
    #         dice_coef,
    #         recall,
    #         precision,
    #         'binary_crossentropy',
    #     ],
    # )

    used_callbacks = []

    # ------------- lr scheduler
    lr_base = 0.01 * (float(batch_size) / 16)
    lr_power = 0.9
    used_callbacks.append(lr_scheduler(epochs, lr_base, lr_power))

    # ------------- tensorboard
    tb = tensorboard('../logs', model_name + '_' + run_name, histogram_freq=0)
    used_callbacks.append(tb)

    # ------------- model checkpoint
    filepath = "weights/" + model_name + '_' + run_name + \
        '.hdf5'  # + "_{epoch:02d}-{categorical_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_categorical_accuracy',
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
        # debug_samples=600
    )

    model = keras.models.load_model('weights/segnet_2018_01_29_08:09.hdf5', custom_objects={
        'dice_coef': dice_coef,
        'precision': precision,
    })

    # ------------- fit!
    model.fit_generator(
        generator=datagen.training_flow(
            labels, batch_size, (target_height, target_width)),
        steps_per_epoch=datagen.steps_per_epoch(batch_size),
        epochs=epochs,
        initial_epoch=16,
        verbose=1,
        validation_data=datagen.validation_flow(
            labels, batch_size, (target_height, target_width)),
        validation_steps=datagen.validation_steps(batch_size),
        callbacks=used_callbacks
    )

    # save final model
    model.save_weights('weights/' + run_name + '_' +
                       str(epochs) + '_finished.h5')


if __name__ == '__main__':
    run_started = strftime("%Y_%m_%d_%H:%M", gmtime())

    try:
        # tries to get dataset path from os environments
        dataset_path = os.environ['DATASETS']
    except:
        dataset_path = '/home/xmlyna06/data/'

    dataset_path = dataset_path + 'gta/'

    images_path = os.path.join(dataset_path, 'images/')
    labels_path = os.path.join(dataset_path, 'labels/')

    # 'mobile_unet
    train(images_path, labels_path, 'segnet', run_started)
