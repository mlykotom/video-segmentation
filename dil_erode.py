import tensorflow as tf
from keras import Input
from keras.layers import Lambda

#
# def tf_dilate_erode(inputs):
#     img_old = inputs[0]
#     img_new = inputs[1]
#
#     # img_old_gray = tf.image.rgb_to_grayscale(img_old)
#     # img_new_gray = tf.image.rgb_to_grayscale(img_new)
#
#     # diff = tf.subtract(img_old_gray, img_new_gray)
#
#     # diff = tf.Constant([1, 1, 1])
#
#     # filter = tf.squeeze(diff, axis=0)
#     # strides = [1, 1, 1]
#     # rates = [1, 1, 1]
#     # dil = tf.nn.dilation2d(np.array(diff), filter, strides, rates, padding='SAME')
#
#     # return dil
#
#

import numpy as np

import cv2

if __name__ == '__main__':
    target_size = 256, 512


    def lol(old, new):
        gray_old = tf.image.rgb_to_grayscale(old)
        gray_new = tf.image.rgb_to_grayscale(new)

        # diff_tf = tf.subtract(gray_old, gray_new)
        diff_tf = tf.subtract(old, new)
        diff_tf = tf.image.convert_image_dtype(diff_tf, dtype=tf.int32)

        ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)).astype('int32')
        filter = np.array([
            ellipse,
            ellipse,
            ellipse
        ])
        filter = np.rollaxis(filter, 0, 3)
        filter = tf.convert_to_tensor(filter, dtype=tf.int32)

        # ellipse_tf = tf.expand_dims(tf.convert_to_tensor(ellipse, dtype=tf.int32), axis=-1)
        # filter = tf.tile(ellipse_tf, [1, 1, 3])

        # print(ellipse_tf, diff_tf, filter)
        # print(filter)

        channels = diff_tf.get_shape().as_list()[-1]

        strides = [1, 1, 1, 1]
        rates = [1, 1, 1, 1]

        erode = tf.nn.erosion2d(diff_tf, filter, strides, rates, 'SAME')
        dilate = tf.nn.dilation2d(erode, filter, strides, rates, 'SAME')

        dilate = tf.cast(dilate, tf.float32) / 256.0
        out = tf.clip_by_value(dilate, 0, 1)
        return out

        # img_old = Input(target_size + (3,), name='img_old')


    # img = Input(target_size + (3,), name='img_new')
    # flo = Input(target_size + (2,), name='flow')

    # what = Lambda(lol)([img_old, img])
    # print(what)

    frames = [
        '../optical_flow/frankfurt/frankfurt_000001_050683_leftImg8bit.png',
        '../optical_flow/frankfurt/frankfurt_000001_050684_leftImg8bit.png',
        '../optical_flow/frankfurt/frankfurt_000001_050685_leftImg8bit.png',
    ]

    # size = (512, 256)
    size = (1024, 512)

    imgs = [
        cv2.resize(cv2.imread(frames[0]), size),
        cv2.resize(cv2.imread(frames[1]), size)
    ]

    # imgs[0] /= 255.0
    # imgs[1] /= 255.0

    cv2.imshow("old img", imgs[0])

    print(imgs[0].size, imgs[1].size)

    arr = [
        np.array([imgs[0] / 255.0]),
        np.array([imgs[1] / 255.0]),
    ]

    cv2.imshow("diff color", imgs[0] - imgs[1])

    with tf.Session() as sess:
        a_old = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        a_new = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        init = tf.global_variables_initializer()
        sess.run(init)

        graph = lol(a_old, a_new)

        out = sess.run(graph, feed_dict={a_old: arr[0], a_new: arr[1]})
        winner = out[0]
        # print(winner)
        print(out.shape, winner.shape, winner.dtype, np.max(out))
        cv2.imshow("opening TF", winner)
        cv2.waitKey()

    # model = Model([img_old, img, flo], x, name='FlowCNN')
