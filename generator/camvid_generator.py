import glob
import itertools
import os
import random

import cv2
import numpy as np
import tensorflow as tf

from base_generator import BaseDataGenerator


class CamVidGenerator(BaseDataGenerator):
    _config = {
        'labels': [
            [64, 128, 64],  # Animal
            [192, 0, 128],  # Archway
            [0, 128, 192],  # Bicyclist
            [0, 128, 64],  # Bridge
            [128, 0, 0],  # Building
            [64, 0, 128],  # Car
            [64, 0, 192],  # CartLuggagePram
            [192, 128, 64],  # Child
            [192, 192, 128],  # Column_Pole
            [64, 64, 128],  # Fence
            [128, 0, 192],  # LaneMkgsDriv
            [192, 0, 64],  # LaneMkgsNonDriv
            [128, 128, 64],  # Misc_Text
            [192, 0, 192],  # MotorcycleScooter
            [128, 64, 64],  # OtherMoving
            [64, 192, 128],  # ParkingBlock
            [64, 64, 0],  # Pedestrian
            [128, 64, 128],  # Road
            [128, 128, 192],  # RoadShoulder
            [0, 0, 192],  # Sidewalk
            [192, 128, 128],  # SignSymbol
            [128, 128, 128],  # Sky
            [64, 128, 192],  # SUVPickupTruck
            [0, 0, 64],  # TrafficCone
            [0, 64, 64],  # TrafficLight
            [192, 64, 128],  # Train
            [128, 128, 0],  # Tree
            [192, 128, 192],  # Truck_Bus
            [64, 0, 64],  # Tunnel
            [192, 192, 0],  # VegetationMisc
            [0, 0, 0],  # Void
            [64, 192, 0],  # Wall
        ],
        # 'weights_file': 'data/pretrained_dilation_camvid.pickle',
        # 'input_shape': (900, 1100, 3),
        # 'output_shape': (66, 91, 11),  # TODO what???
        'mean_pixel': (110.70, 108.77, 105.41),
    }

    def get_config(self):

        return {
            'labels': self._config['labels'],
            'n_classes': len(self._config['labels'])
        }

    def __init__(self, dataset_path, debug_samples=0, validation_split=0.0):
        # validation split cant be full dataset and can't be out of range
        assert 0.0 <= validation_split < 1.0

        # split_index = int((1.0 - validation_split) * len(images))

        # training_img, training_lab = images[:split_index], labels[:split_index]
        # validation_img, validation_lab = images[split_index:], labels[split_index:]

        # self._training_data = zip(training_img, training_lab)
        # self._validation_data = zip(validation_img, validation_lab)
        # self._test_data = # TODO
        dataset_path = os.path.join(dataset_path, 'camvid/')

        super(CamVidGenerator, self).__init__(dataset_path, debug_samples)

    def _fill_split(self, which_set):
        img_path = os.path.join(self._dataset_path, '701_StillsRaw_full/', )
        lab_path = os.path.join(self._dataset_path, 'LabeledApproved_full/', )

        filenames = []
        if which_set == 'train':
            filenames += self._make_pairs(img_path, lab_path, '0016E5_')
            filenames += self._make_pairs(img_path, lab_path, 'Seq05VD_')
            filenames += self._make_pairs(img_path, lab_path, '0006R0_')
        elif which_set == 'val':
            filenames += self._make_pairs(img_path, lab_path, '0001TP_')

        random.shuffle(filenames)

        print('CamVid: ' + which_set + ' ' + str(len(filenames)) + ' files')
        self._data[which_set] = filenames

    def _make_pairs(self, img_path, lab_path, prefix):
        img_files = glob.glob(img_path + prefix + "*.png")
        img_files.sort()
        lab_files = glob.glob(lab_path + prefix + "*.png")
        lab_files.sort()
        return zip(zip(img_files, img_files[1::1]), lab_files)

    def _load_img(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Image %s was not found!" % img_path)
        return img

    def flow(self, type, batch_size, target_size):
        zipped = itertools.cycle(self._data[type])
        while True:
            input1_arr = []
            input2_arr = []
            flow_arr = []
            Y = []

            for _ in range(batch_size):
                (img_old_path, img_new_path), label_path = next(zipped)

                img = cv2.resize(self._load_img(img_old_path), target_size[::-1])
                img2 = cv2.resize(self._load_img(img_new_path), target_size[::-1])
                flow = calc_optical_flow(img, img2, 'dis')

                input1_arr.append(self.normalize(img, target_size))
                input2_arr.append(self.normalize(img2, target_size))
                flow_arr.append(flow)

                seg_tensor = cv2.imread(label_path)
                seg_tensor = self.one_hot_encoding(seg_tensor, target_size)
                Y.append(seg_tensor)

                X = [np.asarray(input1_arr), np.asarray(input2_arr), np.asarray(flow_arr)]

                yield X, Y

    def normalize(self, rgb, target_size):
        # norm = np.zeros(target_size + (3,), np.float32)
        # norm[:, :, 0] = cv2.equalizeHist(rgb[:, :, 0])
        # norm[:, :, 1] = cv2.equalizeHist(rgb[:, :, 1])
        # norm[:, :, 2] = cv2.equalizeHist(rgb[:, :, 2])

        return rgb

        # norm_image = np.zeros_like(rgb, dtype=np.float32)
        # cv2.normalize(rgb, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # return norm_image

    def one_hot_encoding(self, label_img, target_size):
        return BaseDataGenerator.default_one_hot_encoding(label_img, self.labels, target_size)


_disFlow = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOpticalFlow_PRESET_MEDIUM)


def calc_optical_flow(old, new, flow_type):
    old_gray = cv2.cvtColor(old, cv2.COLOR_RGB2GRAY)
    new_gray = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)

    if flow_type == 'dis':
        return _disFlow.calc(old_gray, new_gray, None)
    else:
        return cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def flow_to_bgr(flow, target_size, new_img):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros(target_size + (3,), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def tf_warp(img, flow, H, W):
    def get_pixel_value(img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.
        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*H*W, )
        - y: flattened tensor of shape (B*H*W, )
        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(img, indices)

    #    H = 256
    #    W = 256
    x, y = tf.meshgrid(tf.range(W), tf.range(H))
    x = tf.expand_dims(x, 0)
    x = tf.expand_dims(x, 0)

    y = tf.expand_dims(y, 0)
    y = tf.expand_dims(y, 0)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    grid = tf.concat([x, y], axis=1)
    #    print grid.shape
    flows = grid + flow
    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    x = flows[:, 0, :, :]
    y = flows[:, 1, :, :]
    x0 = x
    y0 = y
    x0 = tf.cast(x0, tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(y0, tf.int32)
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    return out


def calcWarp(img_old, flow, size):
    flow_arr = np.array([np.rollaxis(flow, -1, 0)])

    with tf.Session() as sess:
        a = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        flow_vec = tf.placeholder(tf.float32, shape=[None, 2, None, None])

        init = tf.global_variables_initializer()
        sess.run(init)
        warp_graph = tf_warp(a, flow_arr, size[0], size[1])

        out = sess.run(warp_graph, feed_dict={a: np.array([img_old]), flow_vec: flow_arr})
        out = np.clip(out, 0, 255).astype('uint8')
        winner = out[0].astype('uint8')
        return winner


def get_color_from_label(class_id_image):
    colored_image = np.zeros((class_id_image.shape[0], class_id_image.shape[1], 3), np.uint8)
    for i in range(0, datagen.n_classes):
        colored_image[class_id_image[:, :] == i] = datagen.labels[i]
    return colored_image


def one_hot_to_bgr(label):
    class_scores = label.reshape((target_size[0], target_size[1], datagen.n_classes))
    class_image = np.argmax(class_scores, axis=2)
    colored_class_image = get_color_from_label(class_image)
    colored_class_image = cv2.cvtColor(colored_class_image, cv2.COLOR_RGB2BGR)
    return colored_class_image


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path

        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    else:
        __package__ = ''

    import config

    if os.environ['USER'] == 'mlykotom':
        dataset_path = '/Users/mlykotom/'
    else:
        dataset_path = config.data_path()

    datagen = CamVidGenerator(dataset_path)

    batch_size = 8
    target_size = (360, 480)

    for imgBatch, labelBatch in datagen.flow('train', batch_size, target_size):
        left_img = imgBatch[0][0]
        right_img = imgBatch[1][0]
        optical_flow = imgBatch[2][0]
        label = labelBatch[0]

        flow_bgr = flow_to_bgr(optical_flow, target_size, right_img)

        print(len(imgBatch), imgBatch[0].shape)
        print(left_img.dtype, left_img.shape, right_img.shape, label.shape)
        # exit()

        colored_class_image = one_hot_to_bgr(label)

        winner = calcWarp(left_img, optical_flow, target_size)
        cv2.imshow("winner", winner)

        cv2.imshow("old", left_img)
        cv2.imshow("new", right_img)
        cv2.imshow("flo", flow_bgr)
        cv2.imshow("gt", colored_class_image)
        cv2.imshow("diff", right_img - left_img)
        cv2.waitKey(50)
