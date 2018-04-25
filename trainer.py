import numpy as np

np.random.seed(2018)

import json

import losswise
from keras.callbacks import ModelCheckpoint, LambdaCallback, Callback
from losswise.libs import LosswiseKerasCallback

import config
import utils
from callbacks import SaveLastTrainedEpochCallback, CustomTensorBoard
from generator import *
from models import *


class Trainer:
    train_callbacks = []

    def __init__(self, model_name, dataset_path, target_size, batch_size, n_gpu, debug_samples=0, early_stopping=10, optical_flow_type='dis'):
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

        prev_skip = 0

        # -------------  pick the right model with proper generator
        # -------------------------------------------------------- SEGNET
        if model_name == 'segnet':
            self.datagen = CityscapesGenerator(dataset_path, debug_samples=debug_samples)
            model = SegNet(target_size, self.datagen.n_classes, debug_samples=debug_samples)
        elif model_name == 'segnet_warp':
            self.datagen = CityscapesFlowGenerator(dataset_path, debug_samples=debug_samples)
            model = SegnetWarp0(target_size, self.datagen.n_classes, debug_samples=debug_samples)
        elif model_name == 'segnet_warp1':
            self.datagen = CityscapesFlowGenerator(dataset_path, debug_samples=debug_samples)
            model = SegnetWarp1(target_size, self.datagen.n_classes, debug_samples=debug_samples)
        elif model_name == 'segnet_warp2':
            self.datagen = CityscapesFlowGenerator(dataset_path, debug_samples=debug_samples)
            model = SegnetWarp2(target_size, self.datagen.n_classes, debug_samples=debug_samples)
        elif model_name == 'segnet_warp3':
            self.datagen = CityscapesFlowGenerator(dataset_path, debug_samples=debug_samples)
            model = SegnetWarp3(target_size, self.datagen.n_classes, debug_samples=debug_samples)
        # -------------------------------------------------------- ICNET
        elif model_name == 'icnet':
            self.datagen = CityscapesGeneratorForICNet(dataset_path, debug_samples=debug_samples)
            model = ICNet(target_size, self.datagen.n_classes, debug_samples=debug_samples)
        elif model_name == 'icnet_warp':
            self.datagen = CityscapesFlowGeneratorForICNet(dataset_path, debug_samples=debug_samples, prev_skip=prev_skip, flip_enabled=not is_debug, optical_flow_type=optical_flow_type)
            model = ICNetWarp1(target_size, self.datagen.n_classes, debug_samples=debug_samples)
        elif model_name == 'icnet_warp2':
            self.datagen = CityscapesFlowGeneratorForICNet(dataset_path, debug_samples=debug_samples, prev_skip=prev_skip, flip_enabled=not is_debug, optical_flow_type=optical_flow_type)
            model = ICNetWarp2(target_size, self.datagen.n_classes, debug_samples=debug_samples)
        elif model_name == 'icnet_warp0':
            self.datagen = CityscapesFlowGeneratorForICNet(dataset_path, debug_samples=debug_samples, prev_skip=prev_skip, flip_enabled=not is_debug, optical_flow_type=optical_flow_type)
            model = ICNetWarp0(target_size, self.datagen.n_classes, debug_samples=debug_samples)
        elif model_name == 'icnet_warp012':
            self.datagen = CityscapesFlowGeneratorForICNet(dataset_path, debug_samples=debug_samples, prev_skip=prev_skip, flip_enabled=not is_debug, optical_flow_type=optical_flow_type)
            model = ICNetWarp012(target_size, self.datagen.n_classes, debug_samples=debug_samples)
        elif model_name == 'icnet_warp12':
            self.datagen = CityscapesFlowGeneratorForICNet(dataset_path, debug_samples=debug_samples, prev_skip=prev_skip, flip_enabled=not is_debug, optical_flow_type=optical_flow_type)
            model = ICNetWarp12(target_size, self.datagen.n_classes, debug_samples=debug_samples)
        # -------------------------------------------------------- MOBILE_UNET
        elif model_name == 'mobile_unet':
            self.datagen = CityscapesGenerator(dataset_path, debug_samples=debug_samples)
            model = MobileUNet(target_size, self.datagen.n_classes, debug_samples=debug_samples)
        else:
            self.datagen = CityscapesFlowGenerator(dataset_path, debug_samples=debug_samples, prev_skip=prev_skip, flip_enabled=not is_debug, optical_flow_type=optical_flow_type)
            model = MobileUNetWarp4(target_size, self.datagen.n_classes, debug_samples=debug_samples)

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

        final_path = out_dir + '/' + run_name

        # calculate new folder name (if duplicate)
        if os.path.exists(final_path):
            i = 2
            final_path += '_' + str(i)

            while os.path.exists(final_path):
                i += 1
                final_path = final_path[:-1] + str(i)

        return final_path + name_postfix

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
                self.model.load_model(weights_file)

        return restart_epoch, restart_run_name, batch_size

    def prepare_callbacks(self, run_name, epochs, use_validation_data=False):
        # ------------- tensorboard
        tb = CustomTensorBoard(
            (self.cpu_model if self.n_gpu > 1 else self.model.k),
            self.get_run_path(run_name, '../../logs'),
            self.batch_size,
            histogram_freq=use_validation_data
        )

        self.train_callbacks.append(tb)

        # ------------- model checkpoint
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
        # if not self.is_debug:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=30 if self.is_debug else self._early_stopping,
            verbose=1,
            mode='min'
        )

        self.train_callbacks.append(early_stopping)

        # ------------- lr scheduler
        # lr_base = 0.001  # self.model.optimizer().lr  # * (float(self.batch_size) / 16)
        # lr_power = 0.9
        # self.train_callbacks.append(lr_scheduler(epochs, lr_base, lr_power))

    def fit_model(self, run_name, epochs, restart_training=False):
        # restart_epoch, restart_run_name, batch_size = self.prepare_restarting(restart_training, run_name)
        # if restart_run_name is not None:
        #     run_name = restart_run_name
        # batch_size = batch_size or self.batch_size
        restart_epoch = 0
        batch_size = self.batch_size

        if self.n_gpu > 1 and self.cpu_model is not None:
            # WARNING: multi gpu model not working on version keras 2.1.4, this is workaround
            self.model.k.__setattr__('callback_model', self.cpu_model)

        train_generator = self.datagen.flow('train', batch_size, self.target_size)
        train_steps = self.datagen.steps_per_epoch('train', batch_size)
        val_generator = self.datagen.flow('val', batch_size, self.target_size)
        val_steps = self.datagen.steps_per_epoch('val', batch_size)

        if not self.is_debug:
            # -- shuffle dataset after every epoch
            shuffler = LambdaCallback(on_epoch_end=lambda epoch, logs: self.datagen.shuffle('train'))
            self.train_callbacks.append(shuffler)

        # ------------- losswise dashboard

        losswise_params = {
            'samples': train_steps,
            'batch_size': self.batch_size,
            'model': self.model.name,
            'train_data': {
                'length': self.datagen.data_length('train'),
                'steps': train_steps,
            },
            'epochs': epochs,
            'n_gpus': self.n_gpu,
        }

        losswise_params.update(self.model.params())

        run_name = run_name + 'b%d_lr=%f_dec=%f' % (self.batch_size,
                                                    losswise_params['optimizer']['lr'],
                                                    losswise_params['optimizer']['decay'])

        losswise_callback = LosswiseKerasCallback(
            tag=self.model.name + '|' + run_name,
            params=losswise_params,
            display_interval=2
        )
        # self.train_callbacks.append(losswise_callback)

        # class LosswiseKerasImageCallback(Callback):
        #     def __init__(self, session, train_generator):
        #         self.session = session
        #         self.train_generator = train_generator
        #         super(LosswiseKerasImageCallback, self).__init__()
        #
        #     def get_activations(self, model, layer, X_batch):
        #         get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output, ])
        #         activations = get_activations([X_batch, 0])
        #         return activations
        #
        #     def layer_to_visualize(self, img_to_visualize, layer, model):
        #         inputs = [K.learning_phase()] + model.inputs
        #         get_activations = K.function(inputs, [layer.output])
        #
        #         def convout1_f(X):
        #             # The [0] is to disable the training phase flag
        #             return _convout1_f([0] + [X])
        #
        #         convolutions = convout1_f(img_to_visualize)
        #         convolutions = np.squeeze(convolutions)
        #         convolutions = np.moveaxis(convolutions, 2, 0)
        #         print ('Shape of conv:', convolutions.shape)
        #
        #         # filters = convolutions.shape[0]
        #         # print("convs", filters)
        #         # n = int(np.ceil(np.sqrt(filters)))
        #         #    print("sqrted", n)
        #         # Visualization of each filter of the layer
        #         # fig = plt.figure(figsize=(32, 32))
        #         # for i in range(filters):
        #         #     ax = fig.add_subplot(n, n, i + 1)
        #         #     ax.imshow(convolutions[i], cmap='rgb')
        #
        #     def on_train_begin(self, logs=None):
        #         self.x = 0
        #         self.image_sequence = self.session.image_sequence(x=self.x, name='Transformed flow')
        #
        #     def on_batch_end(self, batch, logs=None):
        #         self.x += 1
        #
        #     def on_epoch_end(self, epoch, logs=None):
        #         logs = logs or {}
        #
        #         self.model.predict(self.xy.x_train, batch_size=p.bath_size)
        #
        #         self.image_sequence.append()
        #
        # image_losswise = LosswiseKerasImageCallback(
        #     losswise_callback.session.image_sequence,
        #     train_generator
        # )
        #
        # self.train_callbacks.append(image_losswise)

        self.prepare_callbacks(run_name, epochs)

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
            use_multiprocessing=True,
            shuffle=False
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
