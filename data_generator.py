import glob
import itertools
import os

import cv2
import numpy as np

import cityscapes_labels
import config
import utils


class SimpleSegmentationGenerator:

    def __init__(self, images_path, labels_path, validation_split=0.0, debug_samples=0, shuffle=False):

        # validation split cant be full dataset and can't be out of range
        assert 0.0 <= validation_split < 1.0

        train_split = self.get_filenames('train')
        self._training_data = []
        for img_id in train_split:
            img_path = os.path.join(images_path, img_id)
            lab_path = os.path.join(labels_path, img_id)
            self._training_data.append((img_path, lab_path))

        validation_split = self.get_filenames('val')
        self._validation_data = []
        for img_id in validation_split:
            img_path = os.path.join(images_path, img_id)
            lab_path = os.path.join(labels_path, img_id)
            self._validation_data.append((img_path, lab_path))

        # TODO test split

        # sample for debugging
        if debug_samples > 0:
            self._training_data = self._training_data[:debug_samples]
            self._validation_data = self._validation_data[:debug_samples]

        print("training samples %d, validating samples %d" % (len(self._training_data), len(self._validation_data)))

        # if shuffle:
        #     c = list(zip(images, labels))
        #     random.shuffle(c)
        #     images, labels = zip(*c)

        # same amount of files
        # assert len(images) == len(labels)

        # split_index = int((1.0 - validation_split) * len(images))
        #
        # training_img, training_lab = images[:split_index], labels[:split_index]
        # validation_img, validation_lab = images[split_index:], labels[split_index:]

        # self._training_data = zip(training_img, training_lab)
        # self._validation_data = zip(validation_img, validation_lab)
        # self._test_data = # TODO

    @DeprecationWarning
    def get_files_from_paths(self, images_path, labels_path):
        # get images files
        if not isinstance(images_path, list):
            images_path = [images_path]

        images = []
        for path in images_path:
            files = self._get_files_from_path(path)
            images += files

        # get labels files
        if not isinstance(labels_path, list):
            labels_path = [labels_path]

        labels = []
        for path in labels_path:
            files = self._get_files_from_path(path)
            labels += files

    @staticmethod
    def _filter_files(all_files, to_remove):
        def filter_not_wanted(fn):
            for id in to_remove:
                if os.path.basename(fn) == (str(id).zfill(5) + '.png'):
                    return False

            return True

        return filter(filter_not_wanted, all_files)

    @staticmethod
    def get_filenames(which_set):
        """Get file names for this set."""

        import scipy.io

        filenames = []
        split = scipy.io.loadmat(os.path.join('gta_read_mapping', 'split.mat'))
        split = split[which_set + "Ids"]

        # To remove (Files with different size in img and mask)
        # TODO general (this is applied only to GTA)
        to_remove = [1, 2] + [15188, ] + [i for i in range(20803, 20835)] + [i for i in range(20858, 20861)]

        for id in split:
            if id not in to_remove:
                filenames.append(str(id[0]).zfill(5) + '.png')

        print('GTA5: ' + which_set + ' ' + str(len(filenames)) + ' files')
        return filenames

    @staticmethod
    def _get_files_from_path(path):
        assert path[-1] == '/'

        # TODO general (this is applied only to GTA)
        to_remove = [1, 2] + [15188, ] + [i for i in range(20803, 20835)] + [i for i in range(20858, 20861)]

        all_files = glob.glob(path + "*.jpg") + glob.glob(path + "*.png") + glob.glob(path + "*.jpeg")
        files = SimpleSegmentationGenerator._filter_files(all_files, to_remove)

        assert len(files) > 0, 'No files in %s folder!' % path

        files.sort()
        return files

    @staticmethod
    def _steps_per_epoch(length, batch_size, gpu_count):
        return int(np.ceil(length / float(batch_size * gpu_count)))

    def steps_per_epoch(self, batch_size, gpu_count=1):
        """
        From Keras documentation: Total number of steps (batches of samples) to yield from generator before
        declaring one epoch finished and starting the next epoch. It should typically be equal to the number of
        unique samples of your dataset divided by the batch size.
        :param batch_size:
        :param gpu_count:
        :return:
        """
        return self._steps_per_epoch(len(self._training_data), batch_size, gpu_count)

    def validation_steps(self, batch_size, gpu_count=1):
        return self._steps_per_epoch(len(self._validation_data), batch_size, gpu_count)

    @staticmethod
    def normalize(rgb, target_size):
        """
        :param rgb:
        :param target_size: (height, width)
        :return:
        """
        rgb = cv2.resize(rgb, (target_size[1], target_size[0]))
        norm = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)

        b = rgb[:, :, 0]
        g = rgb[:, :, 1]
        r = rgb[:, :, 2]

        norm[:, :, 0] = cv2.equalizeHist(b)
        norm[:, :, 1] = cv2.equalizeHist(g)
        norm[:, :, 2] = cv2.equalizeHist(r)

        norm = norm / 255.0

        return norm

    @staticmethod
    def one_hot_encoding(label_img, labels, height, width):
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
        label_img = cv2.resize(label_img, (width, height))

        label_list = []
        for lab in labels:
            label_current = np.all(label_img == lab.color, axis=2).astype('uint8')

            # TODO don't want black boundaries :(

            # getting boundaries!
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            # label_current = cv2.morphologyEx(label_current, cv2.MORPH_GRADIENT, kernel)

            # TODO resize here? because if resizing first, it gets bad boundaries or tiny objects
            # label_current = cv2.resize(label_current, (width, height))

            label_list.append(label_current)

        label_arr = np.array(label_list)

        seg_labels = np.rollaxis(label_arr, 0, 3)
        seg_labels = np.reshape(seg_labels, (label_img.shape[0] * label_img.shape[1], label_arr.shape[0]))

        return seg_labels

    def _flow(self, data, labels, batch_size, target_size):
        """
        :param data:
        :param labels: (cityscapes_labels)
        :param batch_size:
        :param target_size: (height, width)
        :param debug_sample:
        :return:
        """
        zipped = itertools.cycle(data)
        while True:
            X = []
            Y = []

            for _ in range(batch_size):
                img_path, label_path = next(zipped)

                img = cv2.imread(img_path)
                img = self.normalize(img, target_size)
                X.append(img)

                seg_tensor = cv2.imread(label_path)
                seg_tensor = self.one_hot_encoding(seg_tensor, labels, target_size[0], target_size[1])
                Y.append(seg_tensor)

            yield np.array(X), np.array(Y)

    def training_flow(self, labels, batch_size, target_size):
        return self._flow(self._training_data, labels, batch_size, target_size)

    def validation_flow(self, labels, batch_size, target_size):
        return self._flow(self._validation_data, labels, batch_size, target_size)

    def validation_data(self, labels, batch_size, target_size):
        data = []
        labs = []
        for i in range(len(self._validation_data)):
            img, lab = next(self.validation_flow(labels, batch_size, target_size))
            for b in range(batch_size):
                data.append(img[b])
                labs.append(lab[b])

        return np.array(data), np.array(labs)


if __name__ == '__main__':
    dataset_path = config.data_path('gta')

    images_path = os.path.join(dataset_path, 'images/')
    labels_path = os.path.join(dataset_path, 'labels/')

    datagen = SimpleSegmentationGenerator(
        images_path=images_path,
        labels_path=labels_path,
        validation_split=0.2,
        # debug_samples=20
        shuffle=True
    )

    batch_size = 1
    target_size = (360, 648)
    # target_size = (1052, 1914)
    # target_size = (10, 10)
    labels = cityscapes_labels.labels
    n_classes = len(labels)

    i = 3
    for img, label in datagen.training_flow(labels, batch_size, target_size):
        print(i, img.shape, label.shape)

        # lol = labels_path + str(i).zfill(5) + '.png'
        # print(lol)
        # real_gt = cv2.imread(lol)
        # real_gt = cv2.resize(real_gt, (target_size[1], target_size[0]))

        cv2.imshow("normalized", img[0])

        class_scores = label[0]
        class_scores = class_scores.reshape((target_size[0], target_size[1], n_classes))
        class_image = np.argmax(class_scores, axis=2)

        colored_class_image = utils.class_image_to_image(class_image, cityscapes_labels.trainId2label)
        colored_class_image = cv2.cvtColor(colored_class_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("gt", colored_class_image)

        # cv2.imshow("real_gt", real_gt)

        cv2.waitKey()

        i += 1
