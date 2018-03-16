import os

import cv2
import numpy as np

import config
from generator import *
from models import *

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    vid = cv2.VideoCapture("/home/mlyko/data/drive.mp4")

    datagen = CityscapesFlowGenerator(config.data_path())
    model = SegNetWarpDiff(config.target_size(), datagen.n_classes)

    model.k.load_weights('../../weights/city/SegNetWarpDiff/warp_diff.h5')
    model.compile()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('segnet_warp_next.avi', fourcc, 25.0, config.target_size()[::-1])

    print("reading")
    try:
        i = 0
        last_frame = None
        while True:
            ret, frame = vid.read()
            if not ret:
                vid.release()
                print("Released Video Resource")
                break

            if i < 500:
                i += 1
                continue

            frame = cv2.resize(frame, config.target_size()[::-1])
            if last_frame is None:
                last_frame = frame

            flow = datagen.calc_optical_flow(last_frame, frame)

            frame_norm = datagen.normalize(frame, config.target_size())
            last_frame_norm = datagen.normalize(last_frame, config.target_size())

            input = [
                np.array([last_frame_norm]),
                np.array([frame_norm]),
                np.array([flow]),
                np.array([frame_norm - last_frame_norm])
            ]

            prediction = model.k.predict(input, 1, verbose=1)

            colored_class_image = datagen.one_hot_to_bgr(prediction, config.target_size(), datagen.n_classes,
                                                         datagen.labels)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            alpha_blended = 0.5 * colored_class_image + 0.5 * img
            alpha_blended = alpha_blended.astype('uint8')

            out.write(alpha_blended)
            last_frame = frame

            i += 1

    except KeyboardInterrupt:
        # Release the Video Device
        vid.release()
        # Message to be displayed after releasing the device
        print("Released Video Resource")
