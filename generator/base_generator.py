import datetime
import itertools
import time
from abc import ABCMeta, abstractmethod, abstractproperty
from math import ceil

import cv2
import os
import random
import tensorflow as tf
from keras.preprocessing.image import *
import shutil


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


class BaseDataGenerator:
    __metaclass__ = ABCMeta

    _files_loaded = False

    @abstractproperty
    def name(self):
        pass

    def __init__(self, dataset_path, debug_samples=0, flip_enabled=False, rotation=5.0, zoom=0.1, brightness=0.1):
        self._debug_samples = debug_samples
        self.is_augment = debug_samples > 50 or debug_samples == 0
        self._data = {'train': [], 'val': [], 'test': []}
        self.dataset_path = dataset_path
        self.flip_enabled = flip_enabled
        self.zoom = zoom
        self.rotation = rotation
        self.brightness = brightness

        print("--- flip " + str(self.flip_enabled))
        print("--- augmentation " + str(self.is_augment))
        print(dataset_path)

    def load_files(self):
        self._fill_split('train')
        self._fill_split('val')
        self._fill_split('test')

        # sample for debugging
        if self._debug_samples > 0:
            self._data['train'] = self._data['train'][:self._debug_samples]
            half_debug_samples = int(ceil(self._debug_samples * 0.5))
            self._data['val'] = self._data['val'][:half_debug_samples]

        print("training samples %d, validating samples %d, test samples %d" %
              (len(self._data['train']), len(self._data['val']), len(self._data['test'])))

        self._files_loaded = True

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

    def shuffle(self, type):
        print("Cityscapes: shuffling dataset")
        random.shuffle(self._data[type])

    @threadsafe_generator
    def flow(self, type, batch_size, target_size):
        """
        :param type: one of [train,val,test]
        :param batch_size:
        :param target_size:
        :return:
        """
        if not self._files_loaded:
            raise Exception('Files weren\'t loaded first!')

        zipped = itertools.cycle(self._data[type])
        i = 0

        while True:
            X = []
            Y = []

            for _ in range(batch_size):
                img_path, label_path = next(zipped)

                apply_flip = random.randint(0, 1)

                img = self._prep_img(type, img_path, target_size, apply_flip)
                img = self.normalize(img, target_size)
                X.append(img)

                seg_img = self._prep_gt(type, label_path, target_size, apply_flip)
                seg_tensor = self.one_hot_encoding(seg_img, target_size)
                Y.append(seg_tensor)

            i += 1

            x = [np.asarray(X)]
            y = [np.array(Y)]

            yield x, y

    def load_data(self, type, batch_size, target_size):
        """
        Method for loading whole dataset into memory.
        Won't work on large datasets such as Cityscapes because it would tak a lot of RAM to load.
        :param type:
        :param batch_size:
        :param target_size:
        :return:
        """
        data = []
        labs = []
        from utils import print_progress
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

    def data_length(self, type):
        return len(self._data[type])

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
        steps = max(1, int(self.data_length(type) // (batch_size * gpu_count)))
        return steps

    def _load_img(self, img_path):
        """
        Loads image from path or fails with error
        :param img_path:
        :return:
        """
        old_path = img_path
        img_path = self.copy_to_scratch(img_path)
        # try scratch dir (if is found)
        img = cv2.imread(img_path)

        if img is None:
            print("--- %s not found, trying again in a while" % img_path)
            time.sleep(2)
            img = cv2.imread(img_path)
            print("--second attempt (%s) %s" % (img_path, str(img is None)))

        # try old path
        if img is None:
            print("--- reading old path %s" % old_path)
            img = cv2.imread(old_path)
            print("--third attempt (%s) q%s" % (img_path, str(img is None)))

        # raise if file not found
        if img is None:
            raise ValueError("Image %s was not found!" % img_path)
        return img

    def copy_to_scratch(self, img_path):
        if 'SCRATCH' in os.environ:
            scratch_dir = os.environ['SCRATCH'] + '/'
            file_name = os.path.split(img_path)[-1]
            if not os.path.exists(scratch_dir + file_name):
                print("-- copying %s to %s" % (os.environ['SCRATCH'] + '/', file_name))
                shutil.copyfile(img_path, scratch_dir + file_name)
            else:
                img_path = scratch_dir + file_name

        return img_path

    def _prep_img(self, type, img_path, target_size, apply_flip=False):
        img = cv2.resize(self._load_img(img_path), target_size[::-1])

        if self.is_augment and type == 'train':
            if self.brightness:
                factor = 1.0 + abs(random.gauss(mu=0.0, sigma=self.brightness))
                if random.randint(0, 1):
                    factor = 1.0 / factor
                table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
                img = cv2.LUT(img, table)

            if self.flip_enabled and apply_flip:
                img = cv2.flip(img, 1)

        if self.rotation:
            angle = random.gauss(mu=0.0, sigma=self.rotation)
        else:
            angle = 0.0
        if self.zoom:
            scale = random.gauss(mu=1.0, sigma=self.zoom)
        else:
            scale = 1.0

        if self.rotation or self.zoom:
            M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, scale)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        return img

    def normalize(self, rgb, target_size):
        if target_size is not None:
            rgb = cv2.resize(rgb, target_size[::-1])

        norm_image = np.zeros_like(rgb, dtype=np.float32)
        cv2.normalize(rgb, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_image

    def _prep_gt(self, type, label_path, target_size, apply_flip=False):
        seg_img = self._load_img(label_path)

        if self.is_augment and type == 'train':
            if self.flip_enabled and apply_flip:
                seg_img = cv2.flip(seg_img, 1)

        if self.rotation or self.zoom:
            M = cv2.getRotationMatrix2D((seg_img.shape[1] // 2, seg_img.shape[0] // 2), angle, scale)
            seg_img = cv2.warpAffine(seg_img, M, (seg_img.shape[1], seg_img.shape[0]))

        return seg_img

    def one_hot_encoding(self, label_img, target_size):
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
        label_img = cv2.resize(label_img, target_size[::-1], interpolation=cv2.INTER_NEAREST)

        label_list = []

        for lab in self.labels:
            label_current = np.all(label_img == lab, axis=2).astype(np.uint8)
            label_list.append(label_current)

        label_arr = np.array(label_list)
        seg_labels = np.rollaxis(label_arr, 0, 3)

        return seg_labels

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
    optical_flow = None

    def __init__(self, dataset_path, debug_samples=0, flip_enabled=False, rotation=5.0, zoom=0.1, brightness=0.1, optical_flow_type='farn'):
        if not hasattr(self, 'optical_flow_type'):
            self.optical_flow_type = optical_flow_type

        print("-- Optical flow type", self.optical_flow_type)

        if self.optical_flow_type == 'dis':
            print("-- creating optical flow DIS")
            self.optical_flow = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOpticalFlow_PRESET_MEDIUM)
        elif self.optical_flow_type == 'deepflow':
            print("-- creating optical flow DeepFlow")
            self.optical_flow = cv2.optflow.createOptFlow_DeepFlow()
        else:
            print("-- using optical flow Farnenback (default openCV)")
            pass

        super(BaseFlowGenerator, self).__init__(
            dataset_path,
            debug_samples=debug_samples,
            flip_enabled=flip_enabled,
            rotation=rotation,
            zoom=zoom,
            brightness=brightness
        )

    def calc_optical_flow(self, old, new, with_time_difference=False):
        old_gray = cv2.cvtColor(old, cv2.COLOR_RGB2GRAY)
        new_gray = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)

        start = datetime.datetime.now()

        if self.optical_flow is not None:
            flow = self.optical_flow.calc(old_gray, new_gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        end = datetime.datetime.now()
        diff = end - start

        if with_time_difference:
            return flow, diff
        else:
            return flow

    @threadsafe_generator
    def flow(self, type, batch_size, target_size):
        if not self._files_loaded:
            raise Exception('Files weren\'t loaded first!')

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
                flow = self.calc_optical_flow(img2, img)

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
    def calc_warp(img_old, flow, size):
        from models.layers.warp import Warp

        with tf.Session() as sess:
            a = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            flow_vec = tf.placeholder(tf.float32, shape=[None, None, None, 2])

            init = tf.global_variables_initializer()
            sess.run(init)
            warp_graph = Warp.tf_warp(a, flow_vec, size)

            out = sess.run(warp_graph, feed_dict={a: np.array([img_old]), flow_vec: np.array([flow])})
            out = np.clip(out, 0, 1)
            winner = out[0]
            return winner
