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
            model = SegNet(target_size, n_classes, is_debug)
        elif model_name == 'mobile_unet':
            model = MobileUNet(target_size, n_classes, is_debug)
        else:
            raise NotImplemented('Model must be one of [' + ','.join(implemented_models) + ']')

        if n_gpu > 1:
            model.make_multi_gpu(n_gpu)

        return Trainer(model, batch_size, n_gpu, is_debug)

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

    @staticmethod
    def get_gpus():
        """
        :rtype list:
        :return: list of gpus available
        """
        from tensorflow.python.client import device_lib
        return device_lib.list_local_devices()

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
        print("-- Attempt to load saved info %s" % filename)
        epoch = 0
        weights = None

        try:
            with open(filename, 'r') as fp:
                obj = json.load(fp)
                epoch = obj['epoch']
                weights = Trainer.get_save_weights_name(model, epoch)

        except IOError:
            print("Couldn't load %s file" % filename)
        except ValueError:
            print("Couldn't load last epoch (file=%s) JSON values" % filename)

        print("-- Last epoch %d, weights file %s" % (epoch, weights is not None))
        return epoch, weights

    @staticmethod
    def termination_save(model):
        """
        Saves last epoch model
        :param BaseModel model:
        """
        epoch, _ = Trainer.get_last_epoch(model)
        if epoch > 0:
            model.k.save(Trainer.get_save_weights_name(model, epoch), overwrite=True)
            print("===== saving model %s on epoch %d =====" % (model.name, epoch))
        else:
            print("===== SKIPPING saving model because epoch % =====")

    def prepare_restarting(self, is_restart_set):
        """
        :param is_restart_set:
        :return:
        """

        # add save epoch to json callback
        save_epoch_callback = SaveLastTrainedEpochCallback(self.model.name)
        self.train_callbacks.append(save_epoch_callback)

        # register termination save
        atexit.register(self.termination_save, self.model)

        restart_epoch = 0
        if is_restart_set:
            # TODO if restarting, saving to the same run name as previous?
            restart_epoch, weights_file = Trainer.get_last_epoch(self.model)

            if weights_file is not None:
                self.model.load_model(
                    weights_file,
                    custom_objects={
                        'dice_coef': dice_coef,
                        'precision': precision,
                    }
                )

        return restart_epoch

    def prepare_callbacks(self, batch_size, epochs, run_name):
        # ------------- lr scheduler
        lr_base = 0.01 * (float(batch_size) / 16)
        lr_power = 0.9
        self.train_callbacks.append(lr_scheduler(epochs, lr_base, lr_power))

        # ------------- tensorboard
        tb = tensorboard('../logs', self.model.name + '_' + run_name, histogram_freq=0)
        self.train_callbacks.append(tb)

        # ------------- model checkpoint
        filepath = "weights/" + self.model.name + '_' + run_name + '_cat_acc-{categorical_accuracy:.2f}.hdf5'
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='val_categorical_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        self.train_callbacks.append(checkpoint)

    def prepare_data(self, images_path, labels_path, labels, target_size):
        # ------------- data generator
        datagen = data_generator.SimpleSegmentationGenerator(
            images_path=images_path,
            labels_path=labels_path,
            validation_split=0.2,
            debug_samples=20 if self.is_debug else 0
        )

        # TODO find other way than this
        self.train_generator = datagen.training_flow(labels, self.batch_size, target_size)
        self.train_steps = datagen.steps_per_epoch(self.batch_size)
        self.val_generator = datagen.validation_flow(labels, self.batch_size, target_size)
        self.val_steps = datagen.validation_steps(self.batch_size)

    def fit_model(self, run_name='', epochs=100, restart_training=False):

        self.prepare_callbacks(self.batch_size, epochs, run_name)

        restart_epoch = self.prepare_restarting(restart_training)

        self.model.k.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=self.train_steps,
            epochs=epochs,
            initial_epoch=restart_epoch,
            verbose=1,
            validation_data=self.val_generator,
            validation_steps=self.val_steps,
            callbacks=self.train_callbacks
        )


def train(images_path, labels_path, model_name='mobile_unet', run_name='', is_debug=False, restart_training=False,
          n_gpu=1):
    # target_size = 360, 648
    target_size = 384, 640
    labels = cityscapes_labels.labels
    n_classes = len(labels)
    batch_size = 2
    epochs = 200

    trainer = Trainer.from_impl(model_name, target_size, n_classes, batch_size, is_debug, n_gpu)
    model = trainer.compile_model()

    trainer.prepare_data(images_path, labels_path, labels, target_size)

    # train model
    trainer.fit_model(
        run_name=run_name,
        epochs=epochs,
        restart_training=restart_training
    )

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

        parser.add_argument('-g', '--gpus',
                            help='Number of GPUs used for training',
                            default=1)
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
    print("GPUs number", args.gpus)
    print("---------------")

    try:
        train(images_path, labels_path, model_name, run_started,
              is_debug=args.debug,
              restart_training=args.restart,
              n_gpu=int(args.gpus))
    except KeyboardInterrupt:
        print("Keyboard interrupted")
