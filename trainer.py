import json

from keras.callbacks import ModelCheckpoint

import config
import metrics
from callbacks import SaveLastTrainedEpochCallback, tensorboard
from generator import *
from models import *


class Trainer:
    train_callbacks = []

    def __init__(self, model_name, dataset_path, target_size, batch_size, n_gpu, debug_samples=0):
        is_debug = debug_samples > 0
        self.is_debug = is_debug
        self.n_gpu = n_gpu
        self.batch_size = batch_size * n_gpu
        self.target_size = target_size
        print("-- Number of GPUs used %d" % self.n_gpu)
        print("-- Batch size (on all GPUs) %d" % self.batch_size)

        # ------------- data generator
        # self.datagen = GTAGenerator(dataset_path, debug_samples=debug_samples)
        # self.datagen = CamVidGenerator(dataset_path, debug_samples=debug_samples)
        # self.datagen = CamVidFlowGenerator(dataset_path, debug_samples=debug_samples)
        self.datagen = CityscapesGenerator(dataset_path, debug_samples=debug_samples)
        # self.datagen = CityscapesFlowGenerator(dataset_path, debug_samples=debug_samples)

        # -------------  pick the right model
        # if model_name == 'segnet':
        #     model = SegNet(target_size, self.datagen.n_classes, is_debug=is_debug)
        # elif model_name == 'app_mobnet':
        #     model = MobileNetUnet(target_size, self.datagen.n_classes, is_debug=is_debug)
        # elif model_name == 'mobile_unet':
        #     model = MobileUNet(target_size, self.datagen.n_classes, is_debug=is_debug)
        # else:
        #     raise NotImplemented('Unknown model type')

        model = SegNet(target_size, self.datagen.n_classes, is_debug)
        # model = SegNetWarp(target_size, self.datagen.n_classes, is_debug=is_debug)

        # -------------  set multi gpu model
        self.model = model
        self.cpu_model = None
        if n_gpu > 1:
            self.cpu_model = model.k
            model.make_multi_gpu(n_gpu)

    @staticmethod
    def get_save_checkpoint_name(model):
        return './checkpoint/' + ('debug/' if model.is_debug else '') + model.name + '_save_epoch.h5'

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
    def get_run_path(model, run_name):
        return ('debug/' if model.is_debug else '') + model.name + '_' + run_name

    @staticmethod
    def get_last_epoch(model):
        """
        :param BaseModel model:
        :return tuple(int, str): last saved  epoch or 0
        """

        filename = SaveLastTrainedEpochCallback.get_model_file_name(model)
        print("-- Attempt to load saved info %s" % filename)
        epoch = 0
        weights = None
        run_name = None
        batch_size = None
        try:
            with open(filename, 'r') as fp:
                obj = json.load(fp)
                epoch = obj['epoch']
                run_name = obj['run_name']
                batch_size = obj['batch_size']
                weights = Trainer.get_save_checkpoint_name(model)

        except IOError:
            print("Couldn't load %s file" % filename)
        except ValueError:
            print("Couldn't load last epoch (file=%s) JSON values" % filename)

        print("-- Last epoch %d, weights file %s" % (epoch, weights is not None))
        return epoch, run_name, weights, batch_size

    def summaries(self):
        print(self.model.summary())
        self.model.plot_model()

    def prepare_restarting(self, is_restart_set, run_name):
        """
        :param run_name:
        :param is_restart_set:
        :return:
        """

        # add save epoch to json callback
        save_epoch_callback = SaveLastTrainedEpochCallback(self.model, run_name, self.batch_size)
        self.train_callbacks.append(save_epoch_callback)

        epoch_save = ModelCheckpoint(
            self.get_save_checkpoint_name(self.model),
            verbose=1,
        )
        self.train_callbacks.append(epoch_save)

        restart_epoch = 0
        restart_run_name = None
        batch_size = None

        if is_restart_set:
            restart_epoch, restart_run_name, weights_file, batch_size = Trainer.get_last_epoch(self.model)

            if weights_file is not None:
                self.model.load_model(
                    weights_file,
                    custom_objects={
                        'dice_coef': metrics.dice_coef,
                        'precision': metrics.precision,
                        'mean_iou': metrics.mean_iou
                    }
                )

        return restart_epoch, restart_run_name, batch_size

    def prepare_callbacks(self, run_name):
        # ------------- lr scheduler
        # lr_base = 0.01 * (float(batch_size) / 16)
        # lr_power = 0.9
        # self.train_callbacks.append(lr_scheduler(epochs, lr_base, lr_power))

        # ------------- tensorboard
        # TODO copy output folder after each epoch to remote server
        tb = tensorboard('../../logs', self.get_run_path(self.model, run_name), histogram_freq=0)
        self.train_callbacks.append(tb)

        # ------------- model checkpoint
        # TODO will not work on PChradis
        filepath = "../../weights/" + self.get_run_path(self.model, run_name) + '.h5'

        checkpoint = ModelCheckpoint(
            filepath,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        self.train_callbacks.append(checkpoint)

    def fit_model(self, run_name='', epochs=100, restart_training=False):
        restart_epoch, restart_run_name, batch_size = self.prepare_restarting(restart_training, run_name)
        if restart_run_name is not None:
            run_name = restart_run_name

        self.prepare_callbacks(run_name)

        if self.n_gpu > 1 and self.cpu_model is not None:
            # WARNING: multi gpu model not working on version keras 2.1.4, this is workaround
            self.model.k.__setattr__('callback_model', self.cpu_model)

        batch_size = batch_size or self.batch_size

        train_generator = self.datagen.flow('train', batch_size, self.target_size)
        train_steps = self.datagen.steps_per_epoch('train', batch_size)
        val_generator = self.datagen.flow('val', batch_size, self.target_size)
        val_steps = self.datagen.steps_per_epoch('val', batch_size)

        self.model.k.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            initial_epoch=restart_epoch,
            verbose=1,
            validation_data=val_generator,
            validation_steps=val_steps,
            callbacks=self.train_callbacks,
            max_queue_size=20,
            use_multiprocessing=True
        )


if __name__ == '__main__':
    trainer = Trainer(
        model_name='mobile_unet',
        dataset_path=config.data_path(),
        target_size=(288, 480),
        batch_size=2,
        n_gpu=1,
        debug_samples=0
    )
