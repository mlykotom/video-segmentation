from abc import ABCMeta, abstractmethod

import keras.utils


class BaseModel:
    __metaclass__ = ABCMeta

    """dictionary of custom objects (as per keras definition)"""
    custom_objects = {}

    def __init__(self, target_size, n_classes, is_debug=False):
        """
        :param tuple target_size: (height, width)
        :param int n_classes: number of classes
        :param bool is_debug: turns off regularization
        """
        self.target_size = target_size
        self.n_classes = n_classes
        self.is_debug = is_debug

        self._model = self._create_model()

    def make_multi_gpu(self, n_gpu):
        self._model = keras.utils.multi_gpu_model(self._model, n_gpu)

    def load_model(self, filepath, custom_objects=None, compile_model=True):
        """
        Loads model from hdf5 file with layers and weights (complete model)
        Custom layers are loaded from model, here just specify custom losses,metrics,etc

        :param str filepath:
        :param dict custom_objects: For custom metrics,losses,etc. Model will load its custom layers
        :param bool compile: True
        :return: self (for convenience)
        """

        custom_objects = custom_objects or {}
        custom_objects = custom_objects.copy()
        custom_objects.update(self.custom_objects)

        self._model = keras.models.load_model(
            filepath=filepath,
            custom_objects=custom_objects,
            compile=compile_model
        )
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
            to_file = 'model_%s_%dx%d.png' % (self.name, self.target_size[0], self.target_size[1])

            keras.utils.plot_model(
                self.k,
                to_file=to_file,
                show_layer_names=True,
                show_shapes=True
            )

    def save_final(self, run_name, last_epoch, to_file=None):
        """

        :param str run_name:
        :param int last_epoch:
        :param str to_file:
        :return:
        """
        if to_file is None:
            to_file = 'weights/%s_%s_(%d)_finished.h5' % (self.name, run_name, last_epoch)

        self.k.save_weights(to_file)
