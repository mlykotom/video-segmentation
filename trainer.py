import atexit
import json

import keras.losses
import keras.metrics
from keras import optimizers
from keras.callbacks import ModelCheckpoint

import cityscapes_labels
import metrics
from callbacks import SaveLastTrainedEpochCallback, tensorboard
from generator import data_generator
from models import *


class Trainer:
    train_callbacks = []

    @staticmethod
    def from_impl(model_name, target_size, n_classes, batch_size=1, is_debug=False, n_gpu=1):
        """
        :param is_debug:
        :param str model_name: one of ['segnet', 'mobile_unet']
        :param tuple target_size: (height, width)
        :param int n_classes:
        :param int n_gpu:
        :return Trainer:
        """

        if model_name == 'segnet':
            model = SegNet(target_size, n_classes, is_debug=is_debug)
        elif model_name == 'app_mobnet':
            model = MobileUNet(target_size, n_classes, is_debug=is_debug)
        elif model_name == 'mobile_unet':
            model = MobileNetUnet(target_size, n_classes, is_debug)
        else:
            raise NotImplemented('Model must be one of [' + ','.join(['segnet', 'mobnet', 'app_mobnet']) + ']')

        if n_gpu > 1:
            model.make_multi_gpu(n_gpu)

        return Trainer(model, batch_size, n_gpu, is_debug)

    @staticmethod
    def get_save_checkpoint_name(model):
        return './checkpoint/' + model.name + ('_d' if model.is_debug else '') + '_save_epoch.h5'

    @staticmethod
    def get_gpus():
        """
        :rtype list:
        :return: list of gpus available
        """
        from tensorflow.python.client import device_lib
        return device_lib.list_local_devices()

    @staticmethod
    def get_save_weights_name(model, epoch):
        return model.name + ('_d' if model.is_debug else '') + '_save_epoch-%d.h5' % epoch

    @staticmethod
    def get_last_epoch(model):
        """
        :param BaseModel model:
        :return tuple(int, str): last saved  epoch or 0
        """

        filename = SaveLastTrainedEpochCallback.get_model_file_name(model.name, model.is_debug)
        print("-- Attempt to load saved info %s" % filename)
        epoch = 0
        weights = None
        run_name = None
        try:
            with open(filename, 'r') as fp:
                obj = json.load(fp)
                epoch = obj['epoch']
                run_name = obj['run_name']
                weights = Trainer.get_save_checkpoint_name(model)

        except IOError:
            print("Couldn't load %s file" % filename)
        except ValueError:
            print("Couldn't load last epoch (file=%s) JSON values" % filename)

        print("-- Last epoch %d, weights file %s" % (epoch, weights is not None))
        return epoch, run_name, weights

    @staticmethod
    def termination_save(model):
        """
        Saves last epoch model
        :param BaseModel model:
        """
        epoch, _, _ = Trainer.get_last_epoch(model)
        if epoch > 0:
            model.k.save(Trainer.get_save_weights_name(model, epoch), overwrite=True)
            print("===== saving model %s on epoch %d =====" % (model.name, epoch))
        else:
            print("===== SKIPPING saving model because epoch % =====")

    @staticmethod
    def get_run_path(model, run_name):
        return ('debug/' if model.is_debug else '') + model.name + '_' + run_name

    def __init__(self, model, batch_size, n_gpu, is_debug=False):
        """

        :param BaseModel model:
        """
        self.model = model
        self.is_debug = is_debug

        self.n_gpu = n_gpu
        self.batch_size = batch_size * n_gpu
        print("-- Number of GPUs used %d" % self.n_gpu)
        print("-- Batch size (on all GPUs) %d" % self.batch_size)

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
            optimizer=optimizers.Adam(),
            metrics=[
                metrics.dice_coef,
                metrics.precision,
                keras.metrics.categorical_accuracy,
                metrics.mean_iou
            ]
        )

        return self.model

    def prepare_restarting(self, is_restart_set, run_name):
        """
        TODO save batch_number in order to have the same values
        :param run_name:
        :param is_restart_set:
        :return:
        """

        # add save epoch to json callback
        save_epoch_callback = SaveLastTrainedEpochCallback(self.model.name, run_name, self.model.is_debug)
        self.train_callbacks.append(save_epoch_callback)

        epoch_save = ModelCheckpoint(
            self.get_save_checkpoint_name(self.model),
            verbose=1,
        )
        self.train_callbacks.append(epoch_save)

        # register termination save
        atexit.register(self.termination_save, self.model)

        restart_epoch = 0
        restart_run_name = None

        if is_restart_set:
            restart_epoch, restart_run_name, weights_file = Trainer.get_last_epoch(self.model)

            if weights_file is not None:
                self.model.load_model(
                    weights_file,
                    custom_objects={
                        'dice_coef': metrics.dice_coef,
                        'precision': metrics.precision,
                        'mean_iou': metrics.mean_iou
                    }
                )

        return restart_epoch, restart_run_name

    def prepare_callbacks(self, batch_size, epochs, run_name):
        # ------------- lr scheduler
        # lr_base = 0.01 * (float(batch_size) / 16)
        # lr_power = 0.9
        # self.train_callbacks.append(lr_scheduler(epochs, lr_base, lr_power))

        # ------------- tensorboard
        # TODO copy output folder after each epoch to remote server
        tb = tensorboard('../../logs', self.get_run_path(self.model, run_name), histogram_freq=0)
        self.train_callbacks.append(tb)

        # ------------- model checkpoint
        # TODO will not work on PChradis or METACENTRUM!
        filepath = "../../weights/" + self.get_run_path(self.model,
                                                        run_name) + '.h5'  # + '_cat_acc-{categorical_accuracy:.2f}'
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        self.train_callbacks.append(checkpoint)

    def prepare_data(self, dataset_path, target_size):
        # ------------- data generator
        datagen = data_generator.GTAGenerator(dataset_path,
                                              debug_samples=20 if self.is_debug else 0)

        # TODO find other way than this
        self.train_generator = datagen.flow('train', self.batch_size, target_size)
        self.train_steps = datagen.steps_per_epoch('train', self.batch_size)
        self.val_generator = datagen.flow('val', self.batch_size, target_size)
        self.val_steps = datagen.steps_per_epoch('val', self.batch_size)

    def fit_model(self, run_name='', epochs=100, restart_training=False):
        restart_epoch, restart_run_name = self.prepare_restarting(restart_training, run_name)
        if restart_run_name is not None:
            run_name = restart_run_name

        self.prepare_callbacks(self.batch_size, epochs, run_name)

        self.model.k.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=self.train_steps,
            epochs=epochs,
            initial_epoch=restart_epoch,
            verbose=1,
            validation_data=self.val_generator,
            validation_steps=self.val_steps,
            callbacks=self.train_callbacks,
            use_multiprocessing=True,
            max_queue_size=100,
        )


if __name__ == '__main__':
    target_size = 288, 480
    model_name = 'mobile_unet'
    labels = cityscapes_labels.labels
    n_classes = len(labels)
    batch_size = 2
    epochs = 200
    is_debug = False
    n_gpu = 1

    trainer = Trainer.from_impl(model_name, target_size, n_classes, batch_size, is_debug, n_gpu)
