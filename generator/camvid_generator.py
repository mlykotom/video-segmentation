import glob
import os
import random

import cv2

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

    @property
    def name(self):
        return 'camvid'

    @property
    def config(self):
        return {
            'labels': self._config['labels'],
            'n_classes': len(self._config['labels'])
        }

    def __init__(self, dataset_path, debug_samples=0):
        dataset_path = os.path.join(dataset_path, 'camvid/')
        super(CamVidGenerator, self).__init__(dataset_path, debug_samples)

    def _fill_split(self, which_set):
        img_path = os.path.join(self.dataset_path, '701_StillsRaw_full/', )
        lab_path = os.path.join(self.dataset_path, 'LabeledApproved_full/', )

        filenames = []
        if which_set == 'train':
            filenames += self._get_files(img_path, lab_path, '0016E5_')
            filenames += self._get_files(img_path, lab_path, 'Seq05VD_')
            filenames += self._get_files(img_path, lab_path, '0006R0_')
        elif which_set == 'val':
            filenames += self._get_files(img_path, lab_path, '0001TP_')

        random.shuffle(filenames)

        print('CamVid: ' + which_set + ' ' + str(len(filenames)) + ' files')
        self._data[which_set] = filenames

    def _get_files(self, img_path, lab_path, prefix):
        img_files = glob.glob(img_path + prefix + "*.png")
        img_files.sort()
        lab_files = glob.glob(lab_path + prefix + "*.png")
        lab_files.sort()
        return zip(img_files, lab_files)


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

    batch_size = 3
    target_size = (360, 480)

    for imgBatch, labelBatch in datagen.flow('train', batch_size, target_size):
        print(len(imgBatch))

        img = imgBatch[0]
        label = labelBatch[0]

        colored_class_image = datagen.one_hot_to_bgr(label, target_size, datagen.n_classes, datagen.labels)

        cv2.imshow("img", img)
        cv2.imshow("gt", colored_class_image)
        cv2.waitKey()
