import os

import cv2
import numpy as np

from base_generator import BaseDataGenerator


class GTAGenerator(BaseDataGenerator):
    def __init__(self, dataset_path, debug_samples=0):
        super(GTAGenerator, self).__init__(dataset_path, debug_samples)

    def get_labels(self):
        import cityscapes_labels
        return [lab.color for lab in cityscapes_labels.labels]

    def get_n_classes(self):
        return len(self.get_labels())

    def normalize(self, rgb, target_size):
        return BaseDataGenerator.default_normalize(rgb, target_size)

    def one_hot_encoding(self, label_img, target_size):
        return BaseDataGenerator.default_one_hot_encoding(label_img, self.get_labels(), target_size)

    def _fill_split(self, which_set):
        split = self._get_filenames(which_set)

        for img_id in split:
            img_path = os.path.join(self._dataset_path, 'images/', img_id)
            lab_path = os.path.join(self._dataset_path, 'labels/', img_id)
            self._data[which_set].append((img_path, lab_path))

    def _get_filenames(self, which_set):
        """Get file names for this set."""

        import scipy.io

        filenames = []
        split = scipy.io.loadmat(os.path.join('./gta_read_mapping', 'split.mat'))
        split = split[which_set + "Ids"]

        # To remove (Files with different size in img and mask)
        # TODO general (this is applied only to GTA)
        to_remove = [1, 2] + [15188, ] + range(20803, 20835) + range(20858, 20861)

        for id in split:
            if id not in to_remove:
                filenames.append(str(id[0]).zfill(5) + '.png')

        print('GTA5: ' + which_set + ' ' + str(len(filenames)) + ' files')
        return filenames


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path

        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    else:
        __package__ = ''

    import cityscapes_labels
    import config
    import utils

    dataset_path = config.data_path('gta')
    images_path = os.path.join(dataset_path, 'images/')
    labels_path = os.path.join(dataset_path, 'labels/')

    datagen = GTAGenerator(dataset_path=dataset_path)

    batch_size = 1
    target_size = (360, 648)
    # target_size = (1052, 1914)
    # target_size = (10, 10)

    i = 3
    for img, label in datagen.flow('val', batch_size, target_size):
        print(i, img.shape, label.shape)

        cv2.imshow("normalized", img[0])

        class_scores = label[0]
        class_scores = class_scores.reshape((target_size[0], target_size[1], datagen.get_n_classes()))
        class_image = np.argmax(class_scores, axis=2)

        colored_class_image = utils.class_image_to_image(class_image, cityscapes_labels.trainId2label)
        colored_class_image = cv2.cvtColor(colored_class_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("gt", colored_class_image)

        cv2.waitKey()

        i += 1
