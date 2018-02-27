import itertools
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np


class BaseDataGenerator:
    __metaclass__ = ABCMeta

    def __init__(self, dataset_path, debug_samples=0):
        self._data = {'train': [], 'val': [], 'test': []}
        self._dataset_path = dataset_path

        self._fill_split('train')
        self._fill_split('val')
        self._fill_split('test')

        # sample for debugging
        if debug_samples > 0:
            self._data['train'] = self._data['train'][:debug_samples]
            self._data['val'] = self._data['val'][:debug_samples]

        # same amount of files
        # TODO
        # assert len(images) == len(labels)

        print("training samples %d, validating samples %d, test samples %d" %
              (len(self._data['train']), len(self._data['val']), len(self._data['test'])))

    @abstractmethod
    def get_labels(self):
        pass

    @abstractmethod
    def get_n_classes(self):
        pass

    @abstractmethod
    def _fill_split(self, which_set):
        """
        :param which_set: test | val | train
        :rtype list:
        """
        pass

    @staticmethod
    def default_normalize(rgb, target_size):
        """
        :param rgb:
        :param target_size: (height, width)
        :return:
        """
        rgb = cv2.resize(rgb, (target_size[1], target_size[0]))
        norm = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)

        b = rgb[:, :, 0]
        g = rgb[:, :, 1]
        r = rgb[:, :, 2]

        norm[:, :, 0] = cv2.equalizeHist(b)
        norm[:, :, 1] = cv2.equalizeHist(g)
        norm[:, :, 2] = cv2.equalizeHist(r)

        norm = norm / 255.0

        return norm

    @staticmethod
    def default_one_hot_encoding(label_img, labels_colors, target_size):
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
        label_img = cv2.resize(label_img, (target_size[1], target_size[0]))

        label_list = []
        # TODO use labels as array with colors
        for lab in labels_colors:
            label_current = np.all(label_img == lab, axis=2).astype('uint8')

            # TODO don't want black boundaries :(

            # getting boundaries!
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            # label_current = cv2.morphologyEx(label_current, cv2.MORPH_GRADIENT, kernel)

            # TODO resize here? because if resizing first, it gets bad boundaries or tiny objects
            # label_current = cv2.resize(label_current, (target_size[1], target_size[0]))

            label_list.append(label_current)

        label_arr = np.array(label_list)

        seg_labels = np.rollaxis(label_arr, 0, 3)
        seg_labels = np.reshape(seg_labels, (label_img.shape[0] * label_img.shape[1], label_arr.shape[0]))

        return seg_labels

    @abstractmethod
    def normalize(self, rgb, target_size):
        pass

    @abstractmethod
    def one_hot_encoding(self, label_img, target_size):
        pass

    def flow(self, type, batch_size, target_size):
        """
        :param type: one of [train,val,test]
        :param batch_size:
        :param target_size:
        :return:
        """

        zipped = itertools.cycle(self._data[type])
        while True:
            X = []
            Y = []

            for _ in range(batch_size):
                img_path, label_path = next(zipped)

                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("Image %s was not found!" % img_path)

                img = self.normalize(img, target_size)
                X.append(img)

                seg_tensor = cv2.imread(label_path)
                seg_tensor = self.one_hot_encoding(seg_tensor, target_size)
                Y.append(seg_tensor)

            yield np.array(X), np.array(Y)

    def load_data(self, type, batch_size, target_size):
        data = []
        labs = []
        from ..utils import print_progress
        # A List of Items
        data_length = len(self._data[type])

        # Initial call to print 0% progress
        print_progress(0, data_length, prefix='Progress:', suffix='Complete', bar_length=50)

        for i in range(data_length):
            img, lab = next(self.flow(type, batch_size, target_size))
            for b in range(batch_size):
                data.append(img[b])
                labs.append(lab[b])

            # Update Progress Bar
            print_progress(i + 1, data_length, prefix='Progress:', suffix='Complete', bar_length=50)

        return np.array(data), np.array(labs)

    def steps_per_epoch(self, type, batch_size, gpu_count=1):
        """
        From Keras documentation: Total number of steps (batches of samples) to yield from generator before
        declaring one epoch finished and starting the next epoch. It should typically be equal to the number of
        unique samples of your dataset divided by the batch size.
        :param type: train | val | test
        :param batch_size:
        :param gpu_count:
        :return:
        """
        return int(np.ceil(len(self._data[type]) / float(batch_size * gpu_count)))
