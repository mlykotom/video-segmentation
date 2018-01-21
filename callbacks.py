# ###############learning rate scheduler####################
import os

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

            if mode is 'progressive_drops':
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

        print('lr: %f' % lr)
        return lr

    return LearningRateScheduler(lr_scheduler)


def tensorboard(save_path, run=None, histogram_freq=10):
    """
    Tensorboard callback
    :param run:
    :param save_path:
    :param histogram_freq:
    :return:
    """
    tensorboard = TensorBoard(
        log_dir=os.path.join(save_path, run),
        histogram_freq=histogram_freq,
        write_graph=True
    )
    return tensorboard
