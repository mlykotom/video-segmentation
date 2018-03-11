import os
import random
import re

import cv2

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
else:
    __package__ = ''

from base_generator import BaseFlowGenerator
from cityscapes_generator import CityscapesGenerator


class CityscapesFlowGenerator(CityscapesGenerator, BaseFlowGenerator):
    def __init__(self, dataset_path, debug_samples=0, how_many_prev=1):
        self._file_pattern = re.compile("(?P<city>[^_]*)_(?:[^_]+)_(?P<frame>[^_]+)_gtFine_color\.png")
        self._how_many_prev = how_many_prev

        super(CityscapesFlowGenerator, self).__init__(dataset_path, debug_samples)

    def _fill_split(self, which_set):
        img_path = os.path.join(self.dataset_path, 'leftImg8bit', which_set, '')
        lab_path = os.path.join(self.dataset_path, 'gtFine', which_set, '')

        # Get file names for this set
        filenames = []
        for root, dirs, files in os.walk(lab_path):
            for gt_name in files:
                if gt_name.startswith('._') or not (gt_name.endswith('_color.png')):
                    continue

                match = self._file_pattern.match(gt_name)
                if match is None:
                    print("skipping path %s" % path)
                    continue

                match_dict = match.groupdict()
                frame_i = int(match_dict['frame'])

                img_name = os.path.join(img_path, match_dict['city'], gt_name)
                i_batch = []
                for i in range(frame_i - self._how_many_prev, frame_i):
                    frame_str = str(i).zfill(6)
                    name_i = img_name \
                        .replace(match_dict['frame'], frame_str) \
                        .replace("gtFine_color", "leftImg8bit")
                    i_batch.append(name_i)

                i_batch.append(img_name.replace("gtFine_color", "leftImg8bit"))

                filenames.append((i_batch, os.path.join(root, gt_name)))

        random.shuffle(filenames)
        print('Cityscapes: ' + which_set + ' ' + str(len(filenames)) + ' files')

        self._data[which_set] = filenames


if __name__ == '__main__':
    import config

    datagen = CityscapesFlowGenerator(config.data_path())

    batch_size = 3
    # target_size = 288, 480
    target_size = 256, 512
    # target_size = 1024, 2048  # orig size

    for imgBatch, labelBatch in datagen.flow('val', batch_size, target_size):
        left_img = imgBatch[0][0]
        right_img = imgBatch[1][0]
        optical_flow = imgBatch[2][0]
        label = labelBatch[0]

        flow_bgr = datagen.flow_to_bgr(optical_flow, target_size)

        print(left_img.dtype, left_img.shape, right_img.shape, label.shape)

        colored_class_image = datagen.one_hot_to_bgr(label, target_size, datagen.n_classes, datagen.labels)

        winner = datagen.calcWarp(left_img, optical_flow, target_size)
        cv2.imshow("winner", winner)

        cv2.imshow("old", left_img)
        cv2.imshow("new", right_img)
        cv2.imshow("flo", flow_bgr)
        cv2.imshow("gt", colored_class_image)
        cv2.imshow("diff", right_img - left_img)
        cv2.waitKey()
