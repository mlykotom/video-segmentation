import os

import cityscapes_labels
import config
from data_generator import SimpleSegmentationGenerator

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

    batch_size = 3
    target_size = 288, 480
    # target_size = (1052, 1914)
    labels = cityscapes_labels.labels
    n_classes = len(labels)

    val_data = datagen.get_data('val', labels, batch_size, target_size)
    print("all", len(val_data[0]), len(val_data[1]))

    print("one", val_data[0][1235].shape, val_data[1][1235].shape)
