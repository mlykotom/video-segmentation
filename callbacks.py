# ###############learning rate scheduler####################
import json

from keras import callbacks
from keras.callbacks import LearningRateScheduler, TensorBoard, Callback
from losswise import Session


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


import keras.backend as K
import numpy as np


class CustomLosswiseKerasCallback(Callback):
    def __init__(self, tag=None, params={}, display_interval=None):
        # model hyper parameters, json serializable Python object
        self.tag = tag
        if not isinstance(params, dict):
            raise RuntimeError("\"params\" argument must be a valid python dictionary")
        self.params_data = params
        self.graph_map = {}
        self.display_interval = display_interval
        super(CustomLosswiseKerasCallback, self).__init__()

    def on_train_begin(self, logs={}):
        if 'epochs' in self.params and 'samples' in self.params and 'batch_size' in self.params:
            self.max_iter = int(self.params['epochs'] * self.params['samples'] / self.params['batch_size'])
        elif 'steps_per_epoch' in self.params and 'epochs' in self.params:
            self.max_iter = self.params['steps_per_epoch'] * self.params['epochs']
        else:
            self.max_iter = None
        self.session = Session(tag=self.tag, max_iter=self.max_iter, params=self.params_data)
        self.metric_list = []
        for metric in self.params['metrics']:
            if not metric.startswith('val_'):
                if metric not in self.metric_list:
                    self.metric_list.append(metric)
        for metric in self.metric_list:
            if 'acc' in metric or 'iou' in metric:  # TODO changed - added iou
                kind = 'max'
            else:
                kind = 'min'
            self.graph_map[metric] = self.session.graph(metric, kind=kind, display_interval=self.display_interval)
        self.x = 0

    def on_epoch_end(self, epoch, logs={}):
        for metric in self.metric_list:
            metric_val = "val_" + metric
            if metric_val in logs:
                data = {metric_val: logs[metric_val]}
                self.graph_map[metric].append(self.x, data)

    def on_batch_end(self, batch, logs={}):
        for metric in self.metric_list:
            data = {metric: logs.get(metric)}
            self.graph_map[metric].append(self.x, data)
        self.x += 1

    def on_train_end(self, logs={}):
        self.session.done()


class CustomTensorBoard(TensorBoard):
    def __init__(self, proper_model, log_dir, batch_size, histogram_freq=0, track_lr=True):
        self._proper_model = proper_model
        self._track_lr = track_lr
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

        if 'out_mean_iou' in logs:
            out_mean_iou = logs['out_mean_iou']
            del logs['out_mean_iou']
            val_out_mean_iou = logs['val_out_mean_iou']
            del logs['val_out_mean_iou']

            logs.update({"mean_iou": out_mean_iou, "val_mean_iou": val_out_mean_iou})

        if self._track_lr:
            # TODO not working on multi gpus :(
            decay = self._proper_model.optimizer.decay
            iterations = self._proper_model.optimizer.iterations
            lr_with_decay = self._proper_model.optimizer.lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
            lr_value = K.eval(lr_with_decay)
            print("--- LR:", lr_value)
            logs.update({"learning_rate": np.array([lr_value])})

            try:
                lc_weights = self._proper_model.get_layer('linear_combination_1').get_weights()
                lc = {
                    "lc_1_w1": np.average(lc_weights[0]),
                    "lc_1_w2": np.average(lc_weights[1])
                }
                logs.update(lc)
                # print("--- LC", lc_weights[0], lc_weights[1])
            except ValueError:
                # THIS LAYER WAS NOT FOUND, just skip
                pass

        super(CustomTensorBoard, self).on_epoch_end(epoch, logs)


class SaveLastTrainedEpochCallback(callbacks.Callback):
    """
    On epoch end saves currently finished epoch to file
    """

    def __init__(self, model, run_name, batch_size, weights_path):
        self.model_name = model.name
        self.is_debug = model.is_debug
        self.run_name = run_name
        self.batch_size = batch_size
        self.weights_path = weights_path
        super(SaveLastTrainedEpochCallback, self).__init__()

    @staticmethod
    def get_model_file_name(model_name, is_debug):
        return './checkpoint/' + model_name + ('_d' if is_debug else '') + '.last_epoch.json'

    def print_learning_rate(self):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("LR: %f" % K.eval(lr_with_decay))

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
                "batch_size": self.batch_size,
                "weights": self.weights_path
            }, fp)
