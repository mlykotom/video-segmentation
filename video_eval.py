import os

import cv2
import numpy as np
from keras import losses, optimizers
from tensorflow.python.client import device_lib

import cityscapes_labels
import config
import utils
from generator import gta_generator
from models import MobileNetUnet

print(device_lib.list_local_devices())

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

vid = cv2.VideoCapture("/home/mlyko/data/drive.mp4")

target_size = 288, 480
labels = cityscapes_labels.labels
n_classes = len(labels)

dataset_path = config.data_path('gta')
images_path = os.path.join(dataset_path, 'images/')
labels_path = os.path.join(dataset_path, 'labels/')

model = MobileNetUnet((target_size[0], target_size[1]), n_classes)

weights = '/home/mlyko/weights/MobileNetUnet_2018_02_20_08:33_cat_acc-0.89.hdf5'
model.k.load_weights(weights)
model.k.compile(
    loss=losses.categorical_crossentropy,
    optimizer=optimizers.Adam(),
)

fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('mobilenet_eval.avi', fourcc, 20.0, (target_size[1], target_size[0]))

print("reading")
try:
    i = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            vid.release()
            print "Released Video Resource"
            break

        if i < 200:
            i += 1
            continue

        # frame = cv2.imread('/home/mlyko/data/gta/images/23333.png')
        # if frame is None:
        #     raise Exception("Image not found")

        frame = cv2.resize(frame, (target_size[1], target_size[0]))
        norm = gta_generator.GTAGenerator.default_normalize(frame, target_size)
        prediction = model.k.predict(np.array([norm]), 1, verbose=1)

        class_scores = prediction.reshape((target_size[0], target_size[1], n_classes))
        class_image = np.argmax(class_scores, axis=2)
        colored_class_image = utils.class_image_to_image(class_image, cityscapes_labels.trainId2label)

        colored_class_image = cv2.cvtColor(colored_class_image, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        alpha_blended = 0.5 * colored_class_image + 0.5 * img
        alpha_blended = alpha_blended.astype('uint8')

        out.write(alpha_blended)
        # cv2.imwrite("eval_video/res_" + str(i) + ".jpg", colored_class_image)

        i += 1
        # print("yolo", frame.shape)
        # cv2.imshow("res?", colored_class_image)
        # cv2.waitKey()
        # print(frame.shape)
        # cv2.imshow("frame", frame)

except KeyboardInterrupt:
    # Release the Video Device
    vid.release()
    # Message to be displayed after releasing the device
    print "Released Video Resource"
