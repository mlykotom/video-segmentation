import glob
import itertools
import os
import random

import cv2
import numpy as np
import tensorflow as tf

from base_generator import one_hot_to_bgr
from camvid_generator import CamVidGenerator


class CamVidFlowGenerator(CamVidGenerator):
    def _fill_split(self, which_set):
        img_path = os.path.join(self.dataset_path, '701_StillsRaw_full/', )
        lab_path = os.path.join(self.dataset_path, 'LabeledApproved_full/', )

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

    def calc_optical_flow(self, old, new, flow_type):
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

                input1 = self.normalize(img, target_size)
                input2 = self.normalize(img2, target_size)

                input1_arr.append(input1)
                input2_arr.append(input2)
                flow_arr.append(flow)

                seg_tensor = cv2.imread(label_path)
                seg_tensor = self.one_hot_encoding(seg_tensor, target_size)
                out_arr.append(seg_tensor)

            x = [np.asarray(input1_arr), np.asarray(input2_arr), np.asarray(flow_arr)]
            y = np.array(out_arr)
            yield x, y

    def normalize(self, rgb, target_size):
        norm_image = np.zeros_like(rgb, dtype=np.float32)
        cv2.normalize(rgb, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_image


def flow_to_bgr(flow, target_size, new_img):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros(target_size + (3,), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


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

    datagen = CamVidFlowGenerator(dataset_path)

    batch_size = 3
    target_size = 288, 480

    for imgBatch, labelBatch in datagen.flow('train', batch_size, target_size):
        left_img = imgBatch[0][0]
        right_img = imgBatch[1][0]
        optical_flow = imgBatch[2][0]
        label = labelBatch[0]

        flow_bgr = flow_to_bgr(optical_flow, target_size, right_img)

        # print(len(imgBatch), imgBatch[0].shape)
        print(left_img.dtype, left_img.shape, right_img.shape, label.shape)
        # exit()

        colored_class_image = one_hot_to_bgr(label, target_size, datagen.n_classes, datagen.labels)

        winner = calcWarp(left_img, optical_flow, target_size)
        cv2.imshow("winner", winner)

        cv2.imshow("old", left_img)
        cv2.imshow("new", right_img)
        cv2.imshow("flo", flow_bgr)
        cv2.imshow("gt", colored_class_image)
        cv2.imshow("diff", right_img - left_img)
        cv2.waitKey(50)
