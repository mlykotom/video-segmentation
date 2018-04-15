from abc import ABCMeta, abstractmethod

import keras.utils
from keras import optimizers


class BaseModel:
    __metaclass__ = ABCMeta

    def __init__(self, target_size, n_classes, is_debug=False, for_training=True):
        """
        :param tuple target_size: (height, width)
        :param int n_classes: number of classes
        :param bool is_debug: turns off regularization
        """
        self.target_size = target_size
        self.n_classes = n_classes
        self.is_debug = is_debug
        self.training_phase = for_training
        self._model = self._create_model()

    def make_multi_gpu(self, n_gpu):
        from keras.utils import multi_gpu_model
        self._model = multi_gpu_model(self._model, n_gpu)

    def load_model(self, filepath, custom_objects=None, compile_model=True):
        """
        Loads model from hdf5 file with layers and weights (complete model)
        Custom layers are loaded from model, here just specify custom losses,metrics,etc

        :param str filepath:
        :param dict custom_objects: For custom metrics,losses,etc. Model will load its custom layers
        :param bool compile: True
        :return: self (for convenience)
        """

        custom_objects = custom_objects.copy() or {}
        custom_objects.update(self.get_custom_objects())

        self._model.load_weights(
            filepath=filepath,
            by_name=True
        )

        if compile_model:
            self.compile()

            # self._model = keras.models.load_model(
        #     filepath=filepath,
        #     custom_objects=custom_objects,
        #     compile=compile_model
        # )
        return self

    @abstractmethod
    def _create_model(self):
        """
        Creates keras model

        :return: keras model (not compiled)
        :rtype: keras.models.Model

        """
        pass

    @property
    def name(self):
        """
        :rtype: str
        :return: name of the model (class name)
        """
        return type(self).__name__

    def summary(self):
        return self._model.summary()

    @property
    def k(self):
        """
        :rtype: keras.models.Model
        :return: keras model
        """
        return self._model

    def plot_model(self, to_file=None):
        """
        Plots mode into PNG file

        :param str to_file:
        """
        if to_file is None:
            to_file = 'plot/model_%s_%dx%d.png' % (self.name, self.target_size[0], self.target_size[1])
            print("Plotting to file " + to_file)

        keras.utils.plot_model(
            self.k,
            to_file=to_file,
            show_layer_names=True,
            show_shapes=True
        )

    def save_final(self, to_file, last_epoch):
        """

        :param str run_name:
        :param int last_epoch:
        :param str to_file:
        :return:
        """

        self.k.save_weights(to_file + '_%d_finished.h5' % last_epoch)

    def metrics(self):
        import metrics
        return [
            # metrics.precision,
            # metrics.recall,
            # metrics.f1_score,
            # keras.metrics.categorical_accuracy,
            metrics.mean_iou
        ]

    def get_custom_objects(self):
        """dictionary of custom objects (as per keras definition)"""
        import metrics
        return {
            # 'dice_coef': metrics.dice_coef,
            # 'precision': metrics.precision,
            # 'recall': metrics.recall,
            # 'f1_score': metrics.f1_score,
            'mean_iou': metrics.mean_iou
        }

    def optimizer_params(self):
        if self.is_debug:
            # return {'lr': 0.0002, 'decay': 0.0991} # for 120 samples
            return {'lr': 0.00031, 'decay': 0.0999}  # for 20 samples
        else:
            return {'lr': 0.001, 'decay': 0.005}  # running on mlyko.can

    def params(self):
        return {
            'optimizer': {
                'name': type(self.optimizer()).__name__,
                'lr': self.optimizer_params()['lr'],
                'decay': self.optimizer_params()['decay'],
            }
        }

    def optimizer(self):
        params = self.optimizer_params()
        return optimizers.Adam(lr=params['lr'], decay=params['decay'])

    def compile(self):
        self._model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=self.optimizer(),
            metrics=self.metrics()
        )
