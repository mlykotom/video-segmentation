import glob
import os
import random

import cv2

from base_generator import BaseFlowGenerator
from camvid_generator import CamVidGenerator


class CamVidFlowGenerator(CamVidGenerator, BaseFlowGenerator):
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

        flow_bgr = datagen.flow_to_bgr(optical_flow, target_size)

        print(left_img.dtype, left_img.shape, right_img.shape, label.shape)

        colored_class_image = datagen.one_hot_to_bgr(label, target_size, datagen.n_classes, datagen.labels)

        winner = datagen.calc_warp(left_img, optical_flow, target_size)
        cv2.imshow("winner", winner)

        cv2.imshow("old", left_img)
        cv2.imshow("new", right_img)
        cv2.imshow("flo", flow_bgr)
        cv2.imshow("gt", colored_class_image)
        cv2.imshow("diff", right_img - left_img)
        cv2.waitKey()
