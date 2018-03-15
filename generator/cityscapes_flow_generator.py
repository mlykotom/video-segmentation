import cv2
import itertools

import numpy as np

from base_generator import BaseFlowGenerator
from cityscapes_generator import CityscapesGenerator


class CityscapesFlowGenerator(CityscapesGenerator, BaseFlowGenerator):
    def __init__(self, dataset_path, debug_samples=0, how_many_prev=1, prev_skip=0, flow_with_diff=False):
        self._flow_with_diff = flow_with_diff
        super(CityscapesFlowGenerator, self).__init__(dataset_path, debug_samples, how_many_prev, prev_skip)

    def flow(self, type, batch_size, target_size):
        zipped = itertools.cycle(self._data[type])
        while True:
            input1_arr = []
            input2_arr = []
            flow_arr = []
            diff_arr = []
            out_arr = []

            for _ in range(batch_size):
                (img_old_path, img_new_path), label_path = next(zipped)

                img = cv2.resize(self._load_img(img_old_path), target_size[::-1])
                img2 = cv2.resize(self._load_img(img_new_path), target_size[::-1])
                flow = self.calc_optical_flow(img, img2, 'dis')

                input1 = self.normalize(img, target_size=None)
                input2 = self.normalize(img2, target_size=None)
                diff = input2 - input1

                input1_arr.append(input1)
                input2_arr.append(input2)
                diff_arr.append(diff)
                flow_arr.append(flow)

                seg_tensor = cv2.imread(label_path)
                seg_tensor = self.one_hot_encoding(seg_tensor, target_size)
                out_arr.append(seg_tensor)

            if self._flow_with_diff:
                x = [
                    np.asarray(input1_arr),
                    np.asarray(input2_arr),
                    np.asarray(flow_arr),
                    np.asarray(diff_arr)
                ]
            else:
                x = [
                    np.asarray(input1_arr),
                    np.asarray(input2_arr),
                    np.asarray(flow_arr)
                ]

            y = np.array(out_arr)
            yield x, y


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path

        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    else:
        __package__ = ''

    import config

    datagen = CityscapesFlowGenerator(config.data_path(), flow_with_diff=True)

    batch_size = 3
    # target_size = 288, 480
    target_size = 256, 512
    # target_size = 1024, 2048  # orig size

    for imgBatch, labelBatch in datagen.flow('val', batch_size, target_size):
        left_img = imgBatch[0][0]
        right_img = imgBatch[1][0]
        optical_flow = imgBatch[2][0]
        diff = imgBatch[3][0]
        label = labelBatch[0]

        flow_bgr = datagen.flow_to_bgr(optical_flow, target_size)

        print(left_img.dtype, left_img.shape, right_img.shape, label.shape, diff.shape)

        colored_class_image = datagen.one_hot_to_bgr(label, target_size, datagen.n_classes, datagen.labels)

        winner = datagen.calcWarp(left_img, optical_flow, target_size)
        cv2.imshow("winner", winner)

        cv2.imshow("old", left_img)
        cv2.imshow("new", right_img)
        cv2.imshow("flo", flow_bgr)
        cv2.imshow("gt", colored_class_image)
        cv2.imshow("diff", diff)
        cv2.waitKey()
