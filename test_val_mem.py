import os

import cityscapes_labels
import config
from generator.gta_generator import GTAGenerator

if __name__ == '__main__':
    dataset_path = config.data_path('gta')

    images_path = os.path.join(dataset_path, 'images/')
    labels_path = os.path.join(dataset_path, 'labels/')

    datagen = GTAGenerator(dataset_path)

    batch_size = 3
    target_size = 288, 480
    # target_size = (1052, 1914)
    labels = cityscapes_labels.labels
    n_classes = len(labels)

    val_data = datagen.load_data('val', batch_size, target_size)
    print("all", len(val_data[0]), len(val_data[1]))

    print("one", val_data[0][1235].shape, val_data[1][1235].shape)
