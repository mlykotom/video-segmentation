import itertools
import os
import random

import cv2
import numpy as np

from base_generator import BaseFlowGenerator, threadsafe_generator
from cityscapes_flow_generator import CityscapesFlowGenerator

optflow_module = not ('WHICH_SERVER' in os.environ) or os.environ['WHICH_SERVER'] != 'metacentrum'


class CityscapesFlowGeneratorForICNet(CityscapesFlowGenerator, BaseFlowGenerator):
    gt_sub = [4, 8, 16]

    @threadsafe_generator
    def flow(self, type, batch_size, target_size):
        if not self._files_loaded:
            raise Exception('Files weren\'t loaded first!')

        zipped = itertools.cycle(self._data[type])
        i = 0

        while True:
            Y = []
            Y2 = []
            Y3 = []

            input1_arr = []
            input2_arr = []
            flow_arr = []

            for _ in range(batch_size):
                (img_old_path, img_new_path), label_path = next(zipped)
                apply_flip = self.flip_enabled and random.randint(0, 1)

                img_old = self._prep_img(type, img_old_path, target_size, apply_flip)
                img_new = self._prep_img(type, img_new_path, target_size, apply_flip)

                # reverse flow
                if optflow_module:
                    # write optical flow to folder and read it from there
                    flo_file = self.dataset_path + 'flow/' + os.path.split(img_old_path)[-1] + '.flo'
                    if os.path.exists(flo_file):
                        flow = cv2.optflow.readOpticalFlow(flo_file)
                    else:
                        flow = self.calc_optical_flow(img_new, img_old)
                        cv2.optflow.writeOpticalFlow(flo_file, flow)
                        print('writing optflow to %s' % flo_file)
                else:
                    flow = self.calc_optical_flow(img_new, img_old)

                flow_arr.append(flow)

                input1 = self.normalize(img_old, target_size=None)
                input1_arr.append(input1)

                input2 = self.normalize(img_new, target_size=None)
                input2_arr.append(input2)

                seg_img = self._prep_gt(type, label_path, target_size, apply_flip)

                seg_tensor = self.one_hot_encoding(seg_img, tuple(a // 4 for a in target_size))
                Y.append(seg_tensor)

                seg_tensor2 = self.one_hot_encoding(seg_img, tuple(a // 8 for a in target_size))
                Y2.append(seg_tensor2)

                seg_tensor3 = self.one_hot_encoding(seg_img, tuple(a // 16 for a in target_size))
                Y3.append(seg_tensor3)

            x = [
                np.asarray(input1_arr),
                np.asarray(input2_arr),
                np.asarray(flow_arr)
            ]

            y = [np.array(Y), np.array(Y2), np.array(Y3)]
            yield x, y
            i += 1


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path

        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    else:
        __package__ = ''

    import config
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    datagen = CityscapesFlowGeneratorForICNet(config.data_path())
    datagen.load_files()

    batch_size = 3
    target_size = 256, 512

    for imgBatch, labelBatch in datagen.flow('val', batch_size, target_size):
        old_img = imgBatch[0][0]
        new_img = imgBatch[1][0]
        optical_flow = imgBatch[2][0]
        label = labelBatch[0][0]

        flow_bgr = datagen.flow_to_bgr(optical_flow, target_size)

        print(old_img.dtype, old_img.shape, new_img.shape, label.shape)

        target_size_14 = tuple(a // 4 for a in target_size)
        colored_class_image = datagen.one_hot_to_bgr(label, target_size_14, datagen.n_classes, datagen.labels)

        winner = datagen.calc_warp(old_img, optical_flow, target_size)
        cv2.imshow("winner", winner)

        cv2.imshow("old", old_img)
        cv2.imshow("new", new_img)
        cv2.imshow("flo", flow_bgr)
        cv2.imshow("gt", colored_class_image)
        cv2.waitKey()
