# ###############learning rate scheduler####################
import json

import keras.backend as K
import numpy as np
from keras import callbacks
from keras.callbacks import LearningRateScheduler, TensorBoard


def lr_scheduler(epochs, lr_base, lr_power):
    """
    Get learning rate scheduler
    :param epochs:
    :param lr_base:
    :param lr_power:
    :return:
    """

    def lr_scheduler(epoch, mode='power_decay'):
        """if lr_dict.has_key(epoch):
            lr = lr_dict[epoch]
            print 'lr: %f' % lr"""

        if mode is 'power_decay':
            # original lr scheduler
            lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
        elif mode is 'exp_decay':
            # exponential decay
            lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
            # adam default lr
            if mode is 'adam':
                lr = 0.001
        elif mode is 'progressive_drops':
            # drops as progression proceeds, good for sgd
            if epoch > 0.9 * epochs:
                lr = 0.0001
            elif epoch > 0.75 * epochs:
                lr = 0.001
            elif epoch > 0.5 * epochs:
                lr = 0.01
            else:
                lr = 0.1
        else:
            raise NotImplemented('lr_scheduler mode must be one of [power_decay, exp_decay]')

        print('----- lr: %f' % lr)
        return lr

    return LearningRateScheduler(lr_scheduler)


def tensorboard(save_path, batch_size, histogram_freq=0):
    """
    Tensorboard callback
    :param run:
    :param save_path:
    :param histogram_freq:
    :return:
    """
    tensorboard = TensorBoard(
        log_dir=save_path,
        batch_size=batch_size,
        histogram_freq=histogram_freq,
        write_grads=True,
        write_images=True,
        write_graph=True
    )
    return tensorboard


import tensorflow as tf


class CustomTensorBoard(TensorBoard):
    def __init__(self, proper_model, log_dir, batch_size, histogram_freq=0):
        self._proper_model = proper_model
        if histogram_freq > 0:
            print("-- Using tensorboard with histograms")

        super(CustomTensorBoard, self).__init__(
            log_dir,
            histogram_freq=histogram_freq,
            batch_size=batch_size,
            write_graph=True,
            write_grads=True,
            write_images=True,
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # TODO not working on multi gpus :(
        # decay = self._proper_model.optimizer.decay
        # iterations = self._proper_model.optimizer.iterations
        # lr_with_decay = self._proper_model.optimizer.lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        # lr_value = K.eval(lr_with_decay)
        # print("--- LR:", lr_value)
        # logs.update({"learning_rate": np.array([lr_value])})
        super(CustomTensorBoard, self).on_epoch_end(epoch, logs)


class TensorBoardWrapper(TensorBoard):
    """
    # TODO not working :(
    Sets the self.validation_data property for use with TensorBoard callback.
    """

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen  # The generator.
        self.nb_steps = nb_steps  # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs=None):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            # print("shapes", ib.shape, tb.shape)

            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * ib.shape[0],) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * tb.shape[0],) + tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)


class SaveLastTrainedEpochCallback(callbacks.Callback):
    """
    On epoch end saves currently finished epoch to file
    """

    def __init__(self, model, run_name, batch_size):
        self.model_name = model.name
        self.is_debug = model.is_debug
        self.run_name = run_name
        self.batch_size = batch_size
        super(SaveLastTrainedEpochCallback, self).__init__()

    @staticmethod
    def get_model_file_name(model_name, is_debug):
        return './checkpoint/' + model_name + ('_d' if is_debug else '') + '.last_epoch.json'

    # def print_learning_rate(self):
    #     lr = self.model.optimizer.lr
    #     decay = self.model.optimizer.decay
    #     iterations = self.model.optimizer.iterations
    #     lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
    #     print("LR: %f" % K.eval(lr_with_decay))

    def on_epoch_end(self, epoch, logs=None):
        """
        Saves last successfully trained epoch
        :param epoch:
        :param logs:
        :return:
        """

        with open(self.get_model_file_name(self.model_name, self.is_debug), 'w') as fp:
            # saves epoch + 1 (so that this is starting next time)
            json.dump({
                "epoch": epoch + 1,
                "run_name": self.run_name,
                "batch_size": self.batch_size
            }, fp)
