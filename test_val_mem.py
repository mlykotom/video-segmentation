import config
from generator.cityscapes_generator import CityscapesGenerator

if __name__ == '__main__':
    dataset_path = config.data_path()

    datagen = CityscapesGenerator(dataset_path)

    batch_size = 3
    target_size = 256, 512
    # target_size = (1052, 1914)

    val_data = datagen.load_data('train', batch_size, target_size)
    print("all", len(val_data[0]), len(val_data[1]))

    print("one", val_data[0][1235].shape, val_data[1][1235].shape)
