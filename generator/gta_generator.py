import os

import cv2

from base_generator import BaseDataGenerator


class GTAGenerator(BaseDataGenerator):
    @property
    def config(self):
        import cityscapes_labels
        labels = [lab.color for lab in cityscapes_labels.labels]
        return {
            'labels': labels,
            'n_classes': len(labels)
        }

    def __init__(self, dataset_path, debug_samples=0):
        dataset_path = os.path.join(dataset_path, 'gta/')
        super(GTAGenerator, self).__init__(dataset_path, debug_samples)

    @property
    def name(self):
        return 'gta'

    def _fill_split(self, which_set):
        split = self._get_filenames(which_set)

        for img_id in split:
            img_path = os.path.join(self.dataset_path, 'images/', img_id)
            lab_path = os.path.join(self.dataset_path, 'labels/', img_id)
            self._data[which_set].append((img_path, lab_path))

    def _get_filenames(self, which_set):
        """Get file names for this set."""

        import scipy.io

        filenames = []
        split = scipy.io.loadmat(os.path.join('./generator/gta_read_mapping', 'split.mat'))
        split = split[which_set + "Ids"]

        # To remove (Files with different size in img and mask)
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

    datagen = GTAGenerator(dataset_path=config.data_path())

    batch_size = 1
    target_size = 256, 512

    i = 3
    for img, label in datagen.flow('val', batch_size, target_size):
        print(i, img.shape, label.shape)

        colored_class_image = datagen.one_hot_to_bgr(label[0], target_size, datagen.n_classes, datagen.labels)

        cv2.imshow("normalized", img[0])
        cv2.imshow("gt", colored_class_image)
        cv2.waitKey()

        i += 1
