import glob

import cv2
import numpy as np

from base_generator import BaseDataGenerator


class CamVidGenerator(BaseDataGenerator):
    def get_labels(self):
        return [
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
        ]
        # return [[128, 0, 0],
        #         [128, 128, 0],
        #         [128, 128, 128],
        #         [64, 0, 128],
        #         [192, 128, 128],
        #         [128, 64, 128],
        #         [64, 64, 0],
        #         [64, 64, 128],
        #         [192, 192, 128],
        #         [0, 0, 192],
        #         [0, 128, 192]]

    def get_n_classes(self):
        return len(self.get_labels())

    config = {
        # 'weights_file': 'data/pretrained_dilation_camvid.pickle',
        # 'input_shape': (900, 1100, 3),
        # 'output_shape': (66, 91, 11),  # TODO what???
        'mean_pixel': (110.70, 108.77, 105.41),
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

        super(CamVidGenerator, self).__init__(dataset_path, debug_samples)

    def _fill_split(self, which_set):
        img_path = os.path.join(self._dataset_path, '701_StillsRaw_full/', )
        lab_path = os.path.join(self._dataset_path, 'LabeledApproved_full/', )

        # if which_set == 'test':
        #     img_files = glob.glob(img_path + '0006R0_' + "*.png")
        #     lab_files = glob.glob(lab_path + '0006R0_' + "*.png")
        if which_set == 'val':
            img_files = glob.glob(img_path + '0001TP_' + "*.png")
            lab_files = glob.glob(lab_path + '0001TP_' + "*.png")
        elif which_set == 'train':
            img_files = glob.glob(img_path + '0016E5_' + "*.png") \
                        + glob.glob(img_path + 'Seq05VD_' + "*.png") \
                        + glob.glob(img_path + '0006R0_' + "*.png")

            lab_files = glob.glob(lab_path + '0016E5_' + "*.png") \
                        + glob.glob(lab_path + 'Seq05VD_' + "*.png") \
                        + glob.glob(lab_path + '0006R0_' + "*.png")
        else:
            img_files = []
            lab_files = []

        assert len(img_files) == len(lab_files)
        img_files.sort()
        lab_files.sort()

        filenames = zip(img_files, lab_files)

        print('CamVid: ' + which_set + ' ' + str(len(filenames)) + ' files')
        self._data[which_set] = filenames

    def normalize(self, rgb, target_size):
        rgb = cv2.resize(rgb, (target_size[1], target_size[0]))
        return rgb
        # return BaseDataGenerator.default_normalize(rgb, target_size)

    def one_hot_encoding(self, label_img, target_size):
        return BaseDataGenerator.default_one_hot_encoding(label_img, self.get_labels(), target_size)


def calc_optical_flow(old, new, flow_type):
    old_gray = cv2.cvtColor(old, cv2.COLOR_RGB2GRAY)
    new_gray = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)

    if flow_type == 'dis':
        disFlow = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOpticalFlow_PRESET_MEDIUM)
        flow = disFlow.calc(old_gray, new_gray, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow


def flow_to_bgr(flow, old):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(old)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path

        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    else:
        __package__ = ''

    import config
    import os

    dataset_path = config.data_path('camvid')
    datagen = CamVidGenerator(dataset_path)

    batch_size = 1
    target_size = (360, 480)


    def get_color_from_label(class_id_image):
        colored_image = np.zeros((class_id_image.shape[0], class_id_image.shape[1], 3), np.uint8)
        for i in range(0, datagen.get_n_classes()):
            colored_image[class_id_image[:, :] == i] = datagen.get_labels()[i]
        return colored_image


    def one_hot_to_bgr(label):
        class_scores = label.reshape((target_size[0], target_size[1], datagen.get_n_classes()))
        class_image = np.argmax(class_scores, axis=2)
        colored_class_image = get_color_from_label(class_image)
        colored_class_image = cv2.cvtColor(colored_class_image, cv2.COLOR_RGB2BGR)
        return colored_class_image


    old_img = None
    for img, label in datagen.flow('train', batch_size, target_size):
        print(img.shape, label.shape)

        colored_class_image = one_hot_to_bgr(label[0])

        if old_img is not None:
            dis_flow = calc_optical_flow(old_img, img[0], 'dis')
            dis_bgr = flow_to_bgr(dis_flow, old_img)
            cv2.imshow("dis_flow", dis_bgr)

            flan_flow = calc_optical_flow(old_img, img[0], 'flan')
            flan_bgr = flow_to_bgr(flan_flow, old_img)
            cv2.imshow("flan", flan_bgr)

        cv2.imshow("normalized", img[0])
        cv2.imshow("gt", colored_class_image)
        cv2.waitKey()

        old_img = img[0]
