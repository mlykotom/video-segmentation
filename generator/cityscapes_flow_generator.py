import cv2

from base_generator import BaseFlowGenerator
from cityscapes_generator import CityscapesGenerator


class CityscapesFlowGenerator(CityscapesGenerator, BaseFlowGenerator):
    def __init__(self, dataset_path, debug_samples=0, how_many_prev=1):
        super(CityscapesFlowGenerator, self).__init__(dataset_path, debug_samples, how_many_prev)


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path

        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    else:
        __package__ = ''

    import config

    datagen = CityscapesFlowGenerator(config.data_path())

    batch_size = 3
    # target_size = 288, 480
    target_size = 256, 512
    # target_size = 1024, 2048  # orig size

    for imgBatch, labelBatch in datagen.flow('val', batch_size, target_size):
        left_img = imgBatch[0][0]
        right_img = imgBatch[1][0]
        optical_flow = imgBatch[2][0]
        label = labelBatch[0]

        flow_bgr = datagen.flow_to_bgr(optical_flow, target_size)

        print(left_img.dtype, left_img.shape, right_img.shape, label.shape)

        colored_class_image = datagen.one_hot_to_bgr(label, target_size, datagen.n_classes, datagen.labels)

        winner = datagen.calcWarp(left_img, optical_flow, target_size)
        cv2.imshow("winner", winner)

        cv2.imshow("old", left_img)
        cv2.imshow("new", right_img)
        cv2.imshow("flo", flow_bgr)
        cv2.imshow("gt", colored_class_image)
        cv2.imshow("diff", right_img - left_img)
        cv2.waitKey()
