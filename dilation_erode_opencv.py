import os

import cv2
import matplotlib
import datetime

# matplotlib.use('TkAgg')

print(matplotlib.get_backend())

import numpy as np

print(os.getcwd())

dataset_path = './data/'

# frames = [
#     dataset_path + '00012.png',
#     dataset_path + '00013.png'
# ]


city_path = './frankfurt/'
target_size = 1024, 512

# frames = [
#     city_path + 'frankfurt_000000_000293_leftImg8bit.png',
#     city_path + 'frankfurt_000000_000294_leftImg8bit.png'
# ]
frames = [
    city_path + 'frankfurt_000001_050683_leftImg8bit.png',
    city_path + 'frankfurt_000001_050684_leftImg8bit.png',
    city_path + 'frankfurt_000001_050685_leftImg8bit.png'
]
# frames = [
#     city_path + 'frankfurt_000000_000575_leftImg8bit.png',
#     city_path + 'frankfurt_000000_000576_leftImg8bit.png'
# ]

old = cv2.imread(frames[0])
print(old.shape)
if old is None:
    raise Exception("File not found")

old = cv2.resize(old, target_size)

print(old.shape)

# h,s,v = cv2.split(old)


frame = cv2.imread(frames[1])
if frame is None:
    raise Exception("File not found")

frame = cv2.resize(frame, target_size)

print(frame.shape)

# cv2.imshow("old", old)
# cv2.imshow("frame", frame)

old_gray = cv2.cvtColor(old, cv2.COLOR_RGB2GRAY)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

# hsv = np.zeros_like(old)
# print(old.shape, hsv.shape)
#
# optical_flow = cv2.optflow.createOptFlow_DeepFlow()
# # optical_flow = cv2.optflow.createOptFlow_SparseToDense()
# # optical_flow = cv2.optflow.createOptFlow_Farneback()
# # optical_flow = cv2.optflow.createOptFlow_Farneback()
# # optical_flow = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOpticalFlow_PRESET_MEDIUM)
#
# times = []
# for i in range(0, 1):  # TODO 10
#     start = datetime.datetime.now()
#     flow = optical_flow.calc(old_gray, frame_gray, None)
#     end = datetime.datetime.now()
#     times.append((end-start).microseconds / 1000.0)
#
# print("optical flow calc", times)
#
# mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
# hsv[..., 0] = ang * 180 / np.pi / 2
# hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
# hsv[..., 2] = 255

# bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# cv2.imshow("diff", old - frame)

# kernel = np.ones((5,5),np.uint8)

# cv2.dilate()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# kernel = np.ones((5, 5))

# old_norm_image = np.zeros_like(old_gray, dtype=np.float32)
# cv2.normalize(old_gray, old_norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# norm_image = np.zeros_like(frame_gray, dtype=np.float32)
# cv2.normalize(frame_gray, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# diff = old_norm_image - norm_image

diff = frame_gray - old_gray

# cv2.imshow("diff", diff)
#
# erosion = cv2.erode(diff, kernel)
# dilation = cv2.dilate(erosion, kernel)
#
# opening = erosion

# norm_image = np.zeros_like(opening, dtype=np.float32)
# cv2.normalize(opening, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# opening = norm_image
# print(opening.dtype)

opening = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(old_gray - frame_gray, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("closing", closing)


cv2.imshow("opening CV", opening)
# cv2.imshow("dense", bgr)
# cv2.imshow("frame", frame)
cv2.waitKey()
