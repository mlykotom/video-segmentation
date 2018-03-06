import cv2
import keras
from keras import optimizers

import config
import metrics
from generator import CamVidFlowGenerator
from generator.base_generator import one_hot_to_bgr
from models import SegNetWarp
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    target_size = 288, 480
    is_debug = True
    datagen = CamVidFlowGenerator(config.data_path())

    model = SegNetWarp(target_size, datagen.n_classes, is_debug=is_debug)
    model.k.load_weights('checkpoint/debug/SegNetWarp_save_epoch.h5')

    # TODO different when debug or not

    model.k.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
        metrics=[
            metrics.dice_coef,
            metrics.precision,
            keras.metrics.categorical_accuracy,
            metrics.mean_iou
        ],
    )

    f, arr = plt.subplots(2, 1)
    f.set_size_inches(20, 20)

    for imgBatch, gtBatch in datagen.flow('train', 1, target_size):
        img = imgBatch[1][0]
        gt = gtBatch[0]
        print(img.shape, gt.shape)

        colored_class_image = one_hot_to_bgr(gt, target_size, datagen.n_classes, datagen.labels)

        # prediction = model.k.predict(np.array([norm]), batch_size, verbose=1)

        cv2.imshow("gt", colored_class_image)

        cv2.waitKey()
