import itertools
from abc import ABCMeta, abstractmethod, abstractproperty

import cv2
import numpy as np
import tensorflow as tf


class BaseDataGenerator:
    __metaclass__ = ABCMeta

    @abstractproperty
    def name(self):
        pass

    def __init__(self, dataset_path, debug_samples=0):
        self._debug_samples = debug_samples
        self._data = {'train': [], 'val': [], 'test': []}
        self.dataset_path = dataset_path

        print(dataset_path)

        self._fill_split('train')
        self._fill_split('val')
        self._fill_split('test')

        # sample for debugging
        if debug_samples > 0:
            self._data['train'] = self._data['train'][:debug_samples]
            self._data['val'] = self._data['val'][:debug_samples]

        print("training samples %d, validating samples %d, test samples %d" %
              (len(self._data['train']), len(self._data['val']), len(self._data['test'])))

    @abstractproperty
    def config(self):
        return {'labels': None, 'n_classes': None}

    @property
    def n_classes(self):
        return self.config['n_classes']

    @property
    def labels(self):
        """
        :rtype list:
        :return: labels colors
        """
        return self.config['labels']

    @abstractmethod
    def _fill_split(self, which_set):
        """
        :param which_set: test | val | train
        :rtype list:
        """
        pass

    def normalize(self, rgb, target_size, equalize_hist=False):
        if target_size is not None:
            rgb = cv2.resize(rgb, target_size[::-1])

        norm_image = np.zeros_like(rgb, dtype=np.float32)

        if equalize_hist:
            norm_image[:, :, 0] = cv2.equalizeHist(rgb[:, :, 0])
            norm_image[:, :, 1] = cv2.equalizeHist(rgb[:, :, 1])
            norm_image[:, :, 2] = cv2.equalizeHist(rgb[:, :, 2])

        cv2.normalize(rgb, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_image

    def one_hot_encoding(self, label_img, target_size):
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
        label_img = cv2.resize(label_img, target_size[::-1], interpolation=cv2.INTER_NEAREST)

        label_list = []
        for lab in self.labels:
            label_current = np.all(label_img == lab, axis=2).astype(np.uint8)
            label_list.append(label_current)

        label_arr = np.array(label_list)

        seg_labels = np.rollaxis(label_arr, 0, 3)
        seg_labels = np.reshape(seg_labels, (label_img.shape[0] * label_img.shape[1], label_arr.shape[0]))

        return seg_labels

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
                if seg_tensor is None:
                    raise ValueError("GT %s was not found!" % label_path)

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

    def _load_img(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Image %s was not found!" % img_path)
        return img

    @staticmethod
    def get_color_from_label(class_id_image, n_classes, labels):
        colored_image = np.zeros((class_id_image.shape[0], class_id_image.shape[1], 3), np.uint8)
        for i in range(0, n_classes):
            colored_image[class_id_image[:, :] == i] = labels[i]
        return colored_image

    @staticmethod
    def one_hot_to_bgr(label, target_size, n_classes, labels, cvt_color=True):
        class_scores = label.reshape(target_size + (n_classes,))
        class_image = np.argmax(class_scores, axis=2)
        colored_class_image = BaseDataGenerator.get_color_from_label(class_image, n_classes, labels)
        if cvt_color:
            colored_class_image = cv2.cvtColor(colored_class_image, cv2.COLOR_RGB2BGR)
        return colored_class_image


class BaseFlowGenerator(BaseDataGenerator):
    __metaclass__ = ABCMeta

    def __init__(self, dataset_path, debug_samples=0):
        self._disFlow = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOpticalFlow_PRESET_MEDIUM)
        super(BaseFlowGenerator, self).__init__(dataset_path, debug_samples)

    def calc_optical_flow(self, old, new, flow_type='dis'):
        old_gray = cv2.cvtColor(old, cv2.COLOR_RGB2GRAY)
        new_gray = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)

        if flow_type == 'dis':
            return self._disFlow.calc(old_gray, new_gray, None)
        else:
            return cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    def flow(self, type, batch_size, target_size):
        zipped = itertools.cycle(self._data[type])
        while True:
            input1_arr = []
            input2_arr = []
            flow_arr = []
            out_arr = []

            for _ in range(batch_size):
                (img_old_path, img_new_path), label_path = next(zipped)

                img = cv2.resize(self._load_img(img_old_path), target_size[::-1])
                img2 = cv2.resize(self._load_img(img_new_path), target_size[::-1])
                flow = self.calc_optical_flow(img, img2, 'dis')

                input1 = self.normalize(img, target_size=None)
                input2 = self.normalize(img2, target_size=None)

                input1_arr.append(input1)
                input2_arr.append(input2)
                flow_arr.append(flow)

                seg_tensor = cv2.imread(label_path)
                seg_tensor = self.one_hot_encoding(seg_tensor, target_size)
                out_arr.append(seg_tensor)

            x = [np.asarray(input1_arr), np.asarray(input2_arr), np.asarray(flow_arr)]
            y = np.array(out_arr)
            yield x, y

    @staticmethod
    def flow_to_bgr(flow, target_size):
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros(target_size + (3,), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    @staticmethod
    def calcWarp(img_old, flow, size):
        from models.layers import tf_warp

        with tf.Session() as sess:
            a = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            flow_vec = tf.placeholder(tf.float32, shape=[None, None, None, 2])

            init = tf.global_variables_initializer()
            sess.run(init)
            warp_graph = tf_warp(a, flow_vec, size)

            out = sess.run(warp_graph, feed_dict={a: np.array([img_old]), flow_vec: np.array([flow])})
            out = np.clip(out, 0, 1)
            winner = out[0]
            return winner
