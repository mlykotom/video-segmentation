import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import random
from generator import cityscapes_labels
import utils
from generator import *

from models import *
from keras import losses, metrics, optimizers
from metrics import precision, dice_coef

target_size = 256, 512

import config


def layer_to_visualize(img_to_visualize, model, layer):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])

    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + X)

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)
    print ('Shape of conv:', convolutions.shape)

    return convolutions


if __name__ == '__main__':

    labels = cityscapes_labels.labels
    n_classes = len(labels)
    batch_size = 2
    epochs = 10

    dataset_path = config.data_path()

    # segnet
    # model = MobileUNetWarp(target_size, n_classes)
    # model = ICNetWarp1(target_size, n_classes)
    model = SegnetWarp2(target_size, n_classes, for_training=True)
    model.compile()
    print(model.summary())

    # model.k.load_weights('/home/mlyko/weights/city/deb/SegNetWarpDiff/20/2320:47b5_lr=0.000050_dec=0.090000.h5')
    # model.k.load_weights('/home/mlyko/weights/city/deb/SegnetWarp0/20/WARPflow_new_old|diff_new_oldb5_lr=0.000070_dec=0.090000.h5')
    # model.k.load_weights('/home/mlyko/weights/city/deb/SegnetWarp0/20/WARP+flo,old,new,diffb5_lr=0.000070_dec=0.090000.h5')
    # model.k.load_weights('/home/mlyko/weights/city/deb/SegnetWarp0/20/WARP+flo,new,old,diffb5_lr=0.000070_dec=0.090000.h5')
    # model.k.load_weights('/home/mlyko/weights/city/deb/SegnetWarp0/20/only_flowb5_lr=0.000070_dec=0.090000.h5')
    # model.k.load_weights('/home/mlyko/weights/city/deb/SegnetWarp0/20/glorot_normalb5_lr=0.000070_dec=0.090000.h5')
    # model.k.load_weights('/home/mlyko/weights/city/deb/SegnetWarp0/20/random_normal_prev0b5_lr=0.000070_dec=0.090000.h5')
    model.k.load_weights('/home/mlyko/weights/city/rel/SegnetWarp2/random_normal_prev0b5_lr=0.000900_dec=0.050000.h5')

    datagen = CityscapesFlowGenerator(dataset_path, debug_samples=20, prev_skip=0)

    x = 0

    for imgBatch, labelBatch in datagen.flow('train', 1, target_size):
        # if x < 80:
        #     x += 5
        #     continue

        imgOld = imgBatch[0]
        imgNew = imgBatch[1][0]
        inpFlow = imgBatch[2][0]

        print("flow", inpFlow.shape)

        # img_diff = layer_to_visualize(imgBatch, model.k, model.k.get_layer('img_diff'), layers)
        transformed_flow = layer_to_visualize(imgBatch, model.k, model.k.get_layer('transformed_flow'))
        transformed_flow_resize = layer_to_visualize(imgBatch, model.k, model.k.get_layer('resize_bilinear_1'))


        cv2.imshow("flow_GT", datagen.flow_to_bgr(inpFlow, target_size))
        cv2.imshow("flow", datagen.flow_to_bgr(transformed_flow, target_size))
        cv2.imshow("flow_small", datagen.flow_to_bgr(transformed_flow_resize, transformed_flow_resize.shape[:-1]))
        cv2.imshow("img", datagen.denormalize(imgNew))

        # print("diff shape", img_diff.shape)
        # cv2.imshow("img_diff", img_diff)
        cv2.waitKey()


