import os
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import itertools

import numpy as np

from base_generator import BaseFlowGenerator
from cityscapes_generator import CityscapesGenerator


class CityscapesFlowGeneratorForICNet(CityscapesGenerator, BaseFlowGenerator):
    def __init__(self, dataset_path, debug_samples=0, how_many_prev=1, prev_skip=0, flip_enabled=False):
        super(CityscapesFlowGeneratorForICNet, self).__init__(
            dataset_path=dataset_path,
            debug_samples=debug_samples,
            how_many_prev=how_many_prev,
            prev_skip=prev_skip,
            flip_enabled=flip_enabled
        )

    def flow(self, type, batch_size, target_size):
        zipped = itertools.cycle(self._data[type])
        i = 0

        while True:
            in_arr = [[]] * 3
            out_arr = [[]] * len(self.gt_sub)
            Y = []
            Y2 = []
            Y3 = []

            input1_arr = []
            input2_arr = []
            flow_arr = []

            for _ in range(batch_size):
                (img_old_path, img_new_path), label_path = next(zipped)

                apply_flip = random.randint(0, 1)

                img_old = self._prep_img(type, img_old_path, target_size, apply_flip)
                img_new = self._prep_img(type, img_new_path, target_size, apply_flip)

                # reverse flow
                flow = self.calc_optical_flow(img_new, img_old, 'dis')

                flow_arr.append(flow)
                in_arr[2].append(flow)

                input1 = self.normalize(img_old, target_size=None)
                in_arr[0].append(input1)
                input1_arr.append(input1)

                input2 = self.normalize(img_new, target_size=None)
                in_arr[1].append(input2)
                input2_arr.append(input2)

                seg_img = self._prep_gt(type, label_path, target_size, apply_flip)

                seg_tensor = self.one_hot_encoding(seg_img, tuple(a // 4 for a in target_size))  # target_size)
                Y.append(seg_tensor)

                seg_tensor2 = self.one_hot_encoding(seg_img, tuple(a // 8 for a in target_size))
                Y2.append(seg_tensor2)

                seg_tensor3 = self.one_hot_encoding(seg_img, tuple(a // 16 for a in target_size))
                Y3.append(seg_tensor3)
                #
                # if self.gt_sub is None:
                #     seg_tensor = self.one_hot_encoding(seg_img, target_size)
                #     out_arr[i].append(seg_tensor)
                # else:
                #     for i, sub in enumerate(self.gt_sub):
                #         subsampled_target_size = tuple(a // sub for a in target_size)
                #         seg_tensor = self.one_hot_encoding(seg_img, subsampled_target_size)
                #         out_arr[i].append(seg_tensor)

            i += 1

            # x = [np.asarray(j) for j in in_arr]
            # y = [np.array(j) for j in out_arr]

            x = [
                np.asarray(input1_arr),
                np.asarray(input2_arr),
                np.asarray(flow_arr)
            ]

            y = [np.array(Y), np.array(Y2), np.array(Y3)]
            yield x, y


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path

        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    else:
        __package__ = ''

    import config

    datagen = CityscapesFlowGeneratorForICNet(config.data_path())

    batch_size = 3
    # target_size = 288, 480
    target_size = 256, 512
    # target_size = 1024, 2048  # orig size

    for imgBatch, labelBatch in datagen.flow('val', batch_size, target_size):
        old_img = imgBatch[0][0]
        new_img = imgBatch[1][0]
        optical_flow = imgBatch[2][0]
        label = labelBatch[0][0]

        flow_bgr = datagen.flow_to_bgr(optical_flow, target_size)

        print(old_img.dtype, old_img.shape, new_img.shape, label.shape)

        target_size_14 = tuple(a // 4 for a in target_size)
        colored_class_image = datagen.one_hot_to_bgr(label, target_size_14, datagen.n_classes, datagen.labels)

        winner = datagen.calcWarp(old_img, optical_flow, target_size)
        cv2.imshow("winner", winner)

        cv2.imshow("old", old_img)
        cv2.imshow("new", new_img)
        cv2.imshow("flo", flow_bgr)
        cv2.imshow("gt", colored_class_image)
        cv2.imshow("diff", new_img - old_img)
        cv2.waitKey()
