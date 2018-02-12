import argparse
import atexit
from time import gmtime, strftime

import keras.models
from keras import optimizers
from keras.callbacks import ModelCheckpoint

import cityscapes_labels
import config
import data_generator
from callbacks import *
from loss import precision, dice_coef
from models import *

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
implemented_models = ['segnet', 'mobile_unet']


class Trainer:
    """
    #TODO multi gpu training
    """

    @staticmethod
    def from_impl(model_name, target_size, n_classes, is_debug=False):
        """
        :param str model_name: one of ['segnet', 'mobile_unet']
        :param tuple target_size: (height, width)
        :param int n_classes:
        :return Trainer:
        """

        if model_name == 'segnet':
            model = SegNet(target_size, n_classes, is_debug)
        elif model_name == 'mobile_unet':
            model = MobileUNet(target_size, n_classes, is_debug=is_debug)
        else:
            raise NotImplemented('Model must be one of [' + ','.join(implemented_models) + ']')

        return Trainer(model, is_debug)

    def __init__(self, model, is_debug=False):
        """

        :param BaseModel model:
        """
        self.model = model
        self.is_debug = is_debug

    def summaries(self):
        self.model.summary()
        self.model.plot_model()

    def compile_model(self):
        # if is_debug:
        #     model.k.compile(
        #         optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
        #         loss=keras.losses.categorical_crossentropy,
        #         metrics=[
        #             dice_coef,
        #             precision,
        #             keras.metrics.categorical_accuracy
        #         ],
        #     )
        # else:
        self.model.k.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=optimizers.Adam(lr=0.001),
            metrics=[
                dice_coef,
                precision,
                keras.metrics.categorical_accuracy
            ]
        )

        return self.model

    @staticmethod
    def get_save_weights_name(model, epoch):
        return model.name + '_save_epoch-%d.h5' % epoch

    @staticmethod
    def get_last_epoch(model):
        """
        :param BaseModel model:
        :return tuple(int, str): last saved  epoch or 0
        """

        filename = SaveLastTrainedEpochCallback.get_model_file_name(model.name)
        with open(filename, 'r') as fp:
            try:
                obj = json.load(fp)
                epoch = obj['epoch']
                weights = Trainer.get_save_weights_name(model, epoch)

            except IOError as e:
                print("File (%s) for loading last epoch not found" % filename)
                epoch = 0
                weights = None

        return epoch, weights

    @staticmethod
    def termination_save(model):
        """
        Saves last epoch model
        :param BaseModel model:
        """
        epoch, _ = Trainer.get_last_epoch(model)
        model.k.save(Trainer.get_save_weights_name(model, epoch), overwrite=True)
        print("===== saving model %s on epoch %d =====" % (model.name, epoch))

    def fit_model(self, train_generator, train_steps, val_generator, val_steps, epochs=100,
                  restart_training=False, callbacks=None):
        """
        """

        if callbacks is None:
            callbacks = []

        # ------------- for restarting
        save_epoch_callback = SaveLastTrainedEpochCallback(self.model.name)
        callbacks.append(save_epoch_callback)
        atexit.register(self.termination_save, self.model)

        restart_epoch = 0
        if restart_training:
            # TODO if restarting, saving to the same run name as previous?
            restart_epoch, weights_file = Trainer.get_last_epoch(self.model)

            self.model.load_model(
                weights_file,
                custom_objects={
                    'dice_coef': dice_coef,
                    'precision': precision,
                })

        self.model.k.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            initial_epoch=restart_epoch,
            verbose=1,
            validation_data=val_generator,
            validation_steps=val_steps,
            callbacks=callbacks
        )


def train(images_path, labels_path, model_name='mobile_unet', run_name='', is_debug=False, restart_training=False):
    # target_size = 360, 648
    target_size = 384, 640
    labels = cityscapes_labels.labels
    n_classes = len(labels)
    batch_size = 2
    epochs = 200

    trainer = Trainer.from_impl(model_name, target_size, n_classes)
    model = trainer.compile_model()

    # ------------- data generator
    datagen = data_generator.SimpleSegmentationGenerator(
        images_path=images_path,
        labels_path=labels_path,
        validation_split=0.2,
        debug_samples=20 if is_debug else 0
    )

    train_generator = datagen.training_flow(labels, batch_size, target_size)
    train_steps = datagen.steps_per_epoch(batch_size)
    val_generator = datagen.validation_flow(labels, batch_size, target_size)
    val_steps = datagen.validation_steps(batch_size)

    # ------------- callbacks
    used_callbacks = []

    # ------------- lr scheduler
    lr_base = 0.01 * (float(batch_size) / 16)
    lr_power = 0.9
    used_callbacks.append(lr_scheduler(epochs, lr_base, lr_power))

    # ------------- tensorboard
    tb = tensorboard('../logs', model.name + '_' + run_name, histogram_freq=0)
    used_callbacks.append(tb)

    # ------------- model checkpoint
    filepath = "weights/" + model.name + '_' + run_name + '_cat_acc-{categorical_accuracy:.2f}.hdf5'
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    used_callbacks.append(checkpoint)

    # train model
    trainer.fit_model(train_generator, train_steps, val_generator, val_steps, epochs, restart_training, used_callbacks)

    # save final model
    model.save_final(run_name, epochs)


if __name__ == '__main__':
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Train model in keras')
        parser.add_argument('-r', '--restart',
                            action='store_true',
                            help='Restarts training from last saved epoch',
                            default=False)

        parser.add_argument('-d', '--debug',
                            action='store_true',
                            help='Just debug training (few images from dataset)',
                            default=False)
        args = parser.parse_args()
        return args

    args = parse_arguments()
    run_started = strftime("%Y_%m_%d_%H:%M", gmtime())

    dataset_path = config.data_path('gta')
    images_path = os.path.join(dataset_path, 'images/')
    labels_path = os.path.join(dataset_path, 'labels/')

    model_name = 'mobile_unet'

    print("---------------")
    print('dataset path', dataset_path)
    print('model', model_name)
    print("is debug", args.debug)
    print("restart training", args.restart)
    print("---------------")

    train(images_path, labels_path, model_name, run_started,
          is_debug=args.debug,
          restart_training=args.restart)
