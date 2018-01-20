import os
import random

import cv2
import numpy as np
from generator import data_generator
from model import segnet

# target_height, target_width = 360, 480
target_height, target_width = 360, 648
n_classes = 27
batch_size = 2
epochs = 10

dataset_path = '/home/xmlyna06/data/gta/'

images_path = os.path.join(dataset_path, 'images/')
labels_path = os.path.join(dataset_path, 'labels/')

model = segnet.get_model(target_height, target_width, n_classes)

model.load_weights('weights/weights__51-0.50.hdf5')

model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer='sgd',
    metrics=["categorical_accuracy"]
)


def vizualize(prediction, target_width, target_height, n_classes):
    pr = prediction.reshape((target_height, target_width, n_classes)).argmax(axis=2)

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]

    seg_img = np.zeros((target_height, target_width, 3))
    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (target_width, target_height))
    return seg_img


image = cv2.imread(images_path + '00008.png')
print(image.shape)

norm = data_generator.SimpleSegmentationGenerator.normalize(image, (target_height, target_width))

for_prediction = np.array([norm])
prediction = model.predict_proba(for_prediction, batch_size, verbose=1)

predicting_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

viz_segmentation = vizualize(prediction, target_width, target_height, n_classes)

cv2.imshow("what", predicting_image)
cv2.imshow("segmented", viz_segmentation)
cv2.waitKey()