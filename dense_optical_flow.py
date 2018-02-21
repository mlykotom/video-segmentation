import os
# matplotlib.use('TkAgg')
import time

import cv2
import matplotlib

print(matplotlib.get_backend())

import numpy as np

print(os.getcwd())

dataset_path = '/home/mlyko/data/gta/images/'

frames = [
    dataset_path + '00012.png',
    dataset_path + '00013.png'
]

old = cv2.imread(frames[0])
if old is None:
    raise Exception("File not found")

old = cv2.resize(old, (957, 526))

print(old.shape)

# h,s,v = cv2.split(old)

hsv = cv2.cvtColor(old, cv2.COLOR_BGR2HSV)
# cv2.imshow("hsv", hsv)

frame = cv2.imread(frames[1])
if frame is None:
    raise Exception("File not found")

frame = cv2.resize(frame, (957, 526))

print(frame.shape)

# cv2.imshow("old", old)
# cv2.imshow("frame", frame)

old_gray = cv2.cvtColor(old, cv2.COLOR_RGB2GRAY)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

hsv = np.zeros_like(old)
hsv[..., 1] = 255

print(old.shape, hsv.shape)

# flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

disFlow = cv2.optflow.createOptFlow_DIS()

start_time = time.time()
flow = disFlow.calc(old_gray, frame_gray, None)
end_time = time.time() - start_time

print("computation ", end_time)

mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("dense", bgr)

cv2.waitKey()
