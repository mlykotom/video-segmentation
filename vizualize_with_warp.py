import cv2
import numpy as np

import config
from generator import CityscapesFlowGenerator
from models import ICNet

if __name__ == '__main__':
    is_debug = True
    dataset_path = config.data_path()
    target_size = config.target_size()
    epochs = 1
    w_path = config.weights_path() + 'city/'
    batch_size = 1
    datagen = CityscapesFlowGenerator(dataset_path, prev_skip=0)

    # model = SegNetWarp(target_size, datagen.n_classes, is_debug=is_debug)
    # model.k.load_weights('checkpoint/debug/SegNetWarp_save_epoch.h5')

    models = [
        (ICNet(target_size, datagen.n_classes, for_training=False), w_path + '/rel/ICNet/b16_lr9e-4_dec=5e-3.h5'),
    ]

    for m, w in models:
        print("loading model %s" % m.name)
        m.compile()
        m.k.load_weights(w, by_name=True)

    for imgBatch, labelBatch in datagen.flow('val', 1, target_size):
        predictions = []
        colored = []

        imgNew = imgBatch[1][0]
        predicting_image = cv2.cvtColor(imgNew, cv2.COLOR_BGR2RGB)

        i = 0
        print(models[i][0].name)
        predictions.append(models[i][0].k.predict(np.array([imgNew]), batch_size, verbose=1))
        colored.append(datagen.one_hot_to_bgr(predictions[i][0], target_size, datagen.n_classes, datagen.old_labels))

        cv2.imshow("predicting", predicting_image)
        # cv2.imshow("gt", datagen.one_hot_to_bgr(labelBatch, target_size, datagen.n_classes, datagen.labels))
        cv2.waitKey()
