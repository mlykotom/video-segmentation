import json

import losswise
from keras.callbacks import ModelCheckpoint
from losswise.libs import LosswiseKerasCallback

import config
import metrics
import utils
from callbacks import SaveLastTrainedEpochCallback, CustomTensorBoard
from generator import *
from models import *


class Trainer:
    train_callbacks = []

    def __init__(self, model_name, dataset_path, target_size, batch_size, n_gpu, debug_samples=0, early_stopping=20):
        is_debug = debug_samples > 0

        if is_debug:
            losswise.set_api_key('EY32N390I')  # api_key for 'mlykotom/dp_debug'
        else:
            losswise.set_api_key('VY1G5AGSO')  # api_key for 'mlykotom/dp_release'

        self.debug_samples = debug_samples
        self.is_debug = is_debug
        self.n_gpu = n_gpu
        self.batch_size = batch_size * n_gpu
        self.target_size = target_size
        self._early_stopping = early_stopping
        print("-- Number of GPUs used %d" % self.n_gpu)
        print("-- Batch size (on all GPUs) %d" % self.batch_size)

        # ------------- data generator
        # self.datagen = GTAGenerator(dataset_path, debug_samples=debug_samples)
        # self.datagen = CamVidGenerator(dataset_path, debug_samples=debug_samples)
        # self.datagen = CamVidFlowGenerator(dataset_path, debug_samples=debug_samples)
        # self.datagen = CityscapesGenerator(dataset_path, debug_samples=debug_samples)

        # -------------  pick the right model
        # if model_name == 'segnet':
        #     model = SegNet(target_size, self.datagen.n_classes, is_debug=is_debug)
        # elif model_name == 'app_mobnet':
        #     model = MobileNetUnet(target_size, self.datagen.n_classes, is_debug=is_debug)
        # elif model_name == 'mobile_unet':
        #     model = MobileUNet(target_size, self.datagen.n_classes, is_debug=is_debug)
        # else:
        #     raise NotImplemented('Unknown model type')

        if model_name == 'segnet':
            self.datagen = CityscapesGenerator(dataset_path, debug_samples=debug_samples)
            model = SegNet(target_size, self.datagen.n_classes, is_debug=is_debug)
        elif model_name == 'segnet_warp':
            self.datagen = CityscapesFlowGenerator(dataset_path, debug_samples=debug_samples, prev_skip=0,
                                                   flow_with_diff=True)
            model = SegNetWarpDiff123(target_size, self.datagen.n_classes, is_debug=is_debug)
        elif model_name == 'app_mobile_unet':
            self.datagen = CityscapesGenerator(dataset_path, debug_samples=debug_samples)
            model = AppMobileNetUnet(target_size, self.datagen.n_classes, is_debug=is_debug)
        elif model_name == 'mobile_unet':
            self.datagen = CityscapesGenerator(dataset_path, debug_samples=debug_samples)
            model = MobileUNet(target_size, self.datagen.n_classes, is_debug=is_debug)
        elif model_name == 'icnet':
            self.datagen = CityscapesGenerator(dataset_path, debug_samples=debug_samples)
            model = ICNet(target_size, self.datagen.n_classes, is_debug=is_debug)
        elif model_name == 'icnet_warp':
            self.datagen = CityscapesFlowGenerator(dataset_path, debug_samples=debug_samples, prev_skip=0,
                                                   flow_with_diff=True, flip_enabled=not is_debug)
            model = ICNetWarp(target_size, self.datagen.n_classes, is_debug=is_debug)
        else:
            self.datagen = CityscapesFlowGenerator(dataset_path, debug_samples=debug_samples, prev_skip=0,
                                                   flow_with_diff=True, flip_enabled=not is_debug)
            model = MobileUNetWarp2(target_size, self.datagen.n_classes, is_debug=is_debug)

        # -------------  set multi gpu model
        self.model = model
        self.cpu_model = None
        if n_gpu > 1:
            self.cpu_model = model.k
            model.make_multi_gpu(n_gpu)

    @staticmethod
    def get_gpus():
        """
        :rtype list:
        :return: list of gpus available
        """
        from tensorflow.python.client import device_lib
        return device_lib.list_local_devices()

    def get_run_path(self, run_name, prefix_dir='', name_postfix=''):
        import os

        """
        #TODO when directory already exist, use new with prefix _2 or _3 ,...
        :param run_name:
        :param prefix_dir:
        :param name_postfix:
        :return:
        """
        out_dir = os.path.join(
            prefix_dir,
            self.datagen.name,
            'deb' if self.model.is_debug else 'rel',
            self.model.name,
            str(self.debug_samples) if self.is_debug else '',
        )

        if not os.path.isfile(out_dir):
            utils.mkdir_recursive(out_dir)

        return out_dir + '/' + run_name + name_postfix

    def get_save_checkpoint_name(self, run_name):
        return self.get_run_path(run_name, './checkpoint/', '_save_epoch.h5')

    def get_last_epoch(self):
        """
        :return tuple(int, str): last saved  epoch or 0
        """

        filename = SaveLastTrainedEpochCallback.get_model_file_name(self.model.name, self.model.is_debug)
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
                weights = self.get_save_checkpoint_name(run_name)

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
            self.get_save_checkpoint_name(run_name),
            verbose=1,
        )
        self.train_callbacks.append(epoch_save)

        restart_epoch = 0
        restart_run_name = None
        batch_size = None

        if is_restart_set:
            restart_epoch, restart_run_name, weights_file, batch_size = self.get_last_epoch()

            if weights_file is not None:
                self.model.load_model(
                    weights_file,
                    custom_objects={
                        # 'dice_coef': metrics.dice_coef,
                        'precision': metrics.precision,
                        'recall': metrics.recall,
                        'f1_score': metrics.f1_score,
                        'mean_iou': metrics.mean_iou
                    }
                )

        return restart_epoch, restart_run_name, batch_size

    def prepare_callbacks(self, run_name, epochs, use_validation_data=False):
        # ------------- tensorboard
        # TODO copy output folder after each epoch to remote server

        tb = CustomTensorBoard(
            (self.cpu_model if self.n_gpu > 1 else self.model.k),
            self.get_run_path(run_name, '../../logs'),
            self.batch_size,
            histogram_freq=use_validation_data
        )

        self.train_callbacks.append(tb)

        # ------------- model checkpoint
        # TODO will not work on PChradis
        filepath = self.get_run_path(run_name, '../../weights/', '.h5')

        checkpoint = ModelCheckpoint(
            filepath,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        self.train_callbacks.append(checkpoint)

        # ------------- early stopping
        # TODO turn on early stopping
        # if not self.is_debug:
        #     early_stopping = keras.callbacks.EarlyStopping(
        #         monitor='val_loss',
        #         min_delta=0,
        #         patience=self._early_stopping,
        #         verbose=1,
        #         mode='min'
        #     )
        #
        #     self.train_callbacks.append(early_stopping)

        # ------------- lr scheduler
        # lr_base = 0.001  # self.model.optimizer().lr  # * (float(self.batch_size) / 16)
        # lr_power = 0.9
        # self.train_callbacks.append(lr_scheduler(epochs, lr_base, lr_power))

    def fit_model(self, run_name, epochs, restart_training=False):
        restart_epoch, restart_run_name, batch_size = self.prepare_restarting(restart_training, run_name)
        if restart_run_name is not None:
            run_name = restart_run_name

        if self.n_gpu > 1 and self.cpu_model is not None:
            # WARNING: multi gpu model not working on version keras 2.1.4, this is workaround
            self.model.k.__setattr__('callback_model', self.cpu_model)

        batch_size = batch_size or self.batch_size

        train_generator = self.datagen.flow('train', batch_size, self.target_size)
        train_steps = self.datagen.steps_per_epoch('train', batch_size)

        # val_data = self.datagen.load_data('val', batch_size, self.target_size)
        val_data = None
        val_generator = self.datagen.flow('val', batch_size, self.target_size)
        val_steps = self.datagen.steps_per_epoch('val', batch_size)

        if not self.is_debug:
            # -- shuffle dataset after every epoch
            shuffler = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.datagen.shuffle('train'))
            self.train_callbacks.append(shuffler)

        # ------------- losswise dashboard

        losswise_params = {
            'batch': self.batch_size,
            'model': self.model.name,
            'train_data': {
                'length': self.datagen.data_length('train'),
                'steps': train_steps,
            },
            'epochs': epochs,
            'n_gpus': self.n_gpu,
        }

        losswise_params.update(self.model.params())

        losswise_callback = LosswiseKerasCallback(
            tag=run_name,
            params=losswise_params
        )
        self.train_callbacks.append(losswise_callback)

        self.prepare_callbacks(run_name, epochs, use_validation_data=val_data is not None)

        self.model.k.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            initial_epoch=restart_epoch,
            verbose=1,
            # validation_data=val_data,
            validation_data=val_generator,
            validation_steps=val_steps,
            callbacks=self.train_callbacks,
            max_queue_size=20,
            use_multiprocessing=True
        )

        # save final model
        self.model.save_final(self.get_run_path(run_name, '../../weights/'), epochs)


if __name__ == '__main__':
    trainer = Trainer(
        model_name='mobile_unet',
        dataset_path=config.data_path(),
        target_size=(288, 480),
        batch_size=2,
        n_gpu=1,
        debug_samples=0
    )
