import cv2
import numpy as np
import tensorflow as tf


# WARNING: this will work on little-endian architectures (eg Intel x86) only!

# warp using scipy
def warp_image(im, flow):
    """
    Use optical flow to warp image to the next
    :param im: image to warp
    :param flow: optical flow
    :return: warped image
    """
    from scipy import interpolate
    image_height = im.shape[0]
    image_width = im.shape[1]
    flow_height = flow.shape[0]
    flow_width = flow.shape[1]
    n = image_height * image_width
    (iy, ix) = np.mgrid[0:image_height, 0:image_width]
    (fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
    fx = fx.astype(np.float64)
    fy = fy.astype(np.float64)
    fx += flow[:, :, 0]
    fy += flow[:, :, 1]
    mask = np.logical_or(fx < 0, fx > flow_width)
    mask = np.logical_or(mask, fy < 0)
    mask = np.logical_or(mask, fy > flow_height)
    fx = np.minimum(np.maximum(fx, 0), flow_width)
    fy = np.minimum(np.maximum(fy, 0), flow_height)
    points = np.concatenate((ix.reshape(n, 1), iy.reshape(n, 1)), axis=1)
    xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n, 1)), axis=1)
    warp = np.zeros((image_height, image_width, im.shape[2]))
    for i in range(im.shape[2]):
        channel = im[:, :, i]
        values = channel.reshape(n, 1)
        new_channel = interpolate.griddata(points, values, xi, method='cubic')
        new_channel = np.reshape(new_channel, [flow_height, flow_width])
        new_channel[mask] = 1
        warp[:, :, i] = new_channel.astype(np.uint8)

    return warp.astype(np.uint8)


def get_flow(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file'
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            print 'Reading %d x %d flo file' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (1, h[0], w[0], 2))
            data2D = np.transpose(data2D, [0, 3, 1, 2])
            return data2D


def tf_warp(img, flow, H, W):
    def get_pixel_value(img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.
        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*H*W, )
        - y: flattened tensor of shape (B*H*W, )
        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(img, indices)

    #    H = 256
    #    W = 256
    x, y = tf.meshgrid(tf.range(W), tf.range(H))
    x = tf.expand_dims(x, 0)
    x = tf.expand_dims(x, 0)

    y = tf.expand_dims(y, 0)
    y = tf.expand_dims(y, 0)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    grid = tf.concat([x, y], axis=1)
    #    print grid.shape
    flows = grid + flow
    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    x = flows[:, 0, :, :]
    y = flows[:, 1, :, :]
    x0 = x
    y0 = y
    x0 = tf.cast(x0, tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(y0, tf.int32)
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    return out


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
        './sequence_files/frankfurt_000000_000575_leftImg8bit.png',
        './sequence_files/frankfurt_000000_000576_leftImg8bit.png',
        # './sequence_files/frankfurt_000000_000293_leftImg8bit.png',
        # './sequence_files/frankfurt_000000_000294_leftImg8bit.png',
    ]

    # size = (512, 256)
    size = (1024, 512)

    imgs = [
        cv2.resize(cv2.imread(frames[0]), size),
        cv2.resize(cv2.imread(frames[1]), size)
    ]

    flow_normal_axis = calc_optical_flow(imgs[0], imgs[1], 'dis')
    flow = np.rollaxis(flow_normal_axis, -1, 0)

    flow_arr = np.array([flow])

    print(imgs[0].dtype)

    print("i shape", imgs[0].shape)
    arrs = [
        np.array([imgs[0] / 255.0]),
        np.array([imgs[1] / 255.0]),
    ]

    print("normal", flow_normal_axis.shape)
    print("xxx", flow_arr.shape)
    # exit()

    with tf.Session() as sess:
        # a = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        # flow_vec = tf.placeholder(tf.float32, shape=[None, None, None, 2])
        #
        # init = tf.global_variables_initializer()
        # sess.run(init)
        # warp_graph = resampler(a, flow_vec)
        #
        # out = sess.run(warp_graph, feed_dict={a: arrs[0], flow_vec: np.array([flow_normal_axis])})
        # out = np.clip(out, 0, 1)
        #
        # cv2.imshow("left", arrs[0][0])
        # cv2.imshow("right", arrs[1][0])
        # cv2.imshow("winner", out[0])
        # cv2.waitKey()

        a = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        flow_vec = tf.placeholder(tf.float32, shape=[None, 2, None, None])
        init = tf.global_variables_initializer()
        sess.run(init)
        warp_graph = tf_warp(a, flow_arr, size[1], size[0])

        out = sess.run(warp_graph, feed_dict={a: arrs[0], flow_vec: flow_arr})
        out = np.clip(out, 0, 1)
        winner = out[0]

        cv2.imshow("winner", winner)
        cv2.imshow("new", imgs[1])
        cv2.imshow("flow", flow_to_bgr(flow_normal_axis, imgs[0]))
        cv2.waitKey()
