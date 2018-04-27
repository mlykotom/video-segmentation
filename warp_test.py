import cv2
import numpy as np
import tensorflow as tf

from models.layers import Warp
from generator.cityscapes_flow_generator import CityscapesFlowGenerator

def flow_to_bgr(flow, old):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(old)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def calc_optical_flow(old, new, flow_type):
    old_gray = cv2.cvtColor(old, cv2.COLOR_RGB2GRAY)
    new_gray = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)

    if flow_type == 'dis':
        disFlow = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOpticalFlow_PRESET_MEDIUM)
        flow = disFlow.calc(old_gray, new_gray, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow


if __name__ == "__main__":
    frames = [
        '../optical_flow/frankfurt/frankfurt_000000_000293_leftImg8bit.png',
        '../optical_flow/frankfurt/frankfurt_000000_000294_leftImg8bit.png',
        # './sequence_files/frankfurt_000000_000293_leftImg8bit.png',
        # './sequence_files/frankfurt_000000_000294_leftImg8bit.png',
    ]

    # datagen = CityscapesFlowGenerator()

    size = (512, 256)
    # size = (1024, 512)

    imgs = [
        cv2.resize(cv2.imread(frames[0]), size),
        cv2.resize(cv2.imread(frames[1]), size)
    ]

    flow = calc_optical_flow(imgs[0], imgs[1], 'dis')
    flow_arr = np.array([flow])

    print(imgs[0].dtype)

    print("i shape", imgs[0].shape)
    arrs = [
        np.array([imgs[0] / 255.0]),
        np.array([imgs[1] / 255.0]),
    ]

    with tf.Session() as sess:
        a = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        flow_vec = tf.placeholder(tf.float32, shape=[None, None, None, 2])
        init = tf.global_variables_initializer()
        sess.run(init)
        warp_graph = Warp.tf_warp(a, flow_arr, size[::-1])

        out = sess.run(warp_graph, feed_dict={a: arrs[1], flow_vec: flow_arr})
        out = np.clip(out, 0, 1)
        winner = out[0]

        cv2.imshow("winner", winner)
        cv2.imshow("old", imgs[0])
        cv2.imshow("new", imgs[1])
        cv2.imshow("flow", flow_to_bgr(flow, imgs[0]))
        cv2.waitKey()
