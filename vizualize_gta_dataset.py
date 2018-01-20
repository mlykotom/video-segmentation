import glob
import random

import cv2
import numpy as np


def imageSegmentationGenerator(images_path, segs_path, n_classes):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "01*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "01*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]
    print("classes", len(colors))



    assert len(images) == len(segmentations)

    for im_fn, seg_fn in zip(images, segmentations):
        assert (im_fn.split('/')[-1] == seg_fn.split('/')[-1])

        img = cv2.imread(im_fn)
        seg = cv2.imread(seg_fn)

        print("img shape", img.shape)

        classes_colors = np.unique(seg)
        print("classes colors", classes_colors)
        seg_img = np.zeros_like(seg)

        for c in range(n_classes):
            seg_img[:, :, 0] += ((seg[:, :, 0] == classes_colors[c]) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((seg[:, :, 0] == classes_colors[c]) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((seg[:, :, 0] == classes_colors[c]) * (colors[c][2])).astype('uint8')


        cv2.imshow("img", img)
        cv2.imshow("segmented", seg_img)
        cv2.waitKey()
        break


images = '/home/xmlyna06/data/gta/images/'
annotations = '/home/xmlyna06/data/gta/labels/'
n_classes = 27
imageSegmentationGenerator(images, annotations, n_classes)
