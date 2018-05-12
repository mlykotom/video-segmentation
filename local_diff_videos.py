import cv2
import numpy as np

video_postfix = 'stuttgart_02.mp4'

video = cv2.VideoCapture("/Users/mlykotom/Downloads/" + video_postfix)
warp_video = cv2.VideoCapture("/Users/mlykotom/Downloads/ICNetWarp2_" + video_postfix + "_drive.avi")
norm_video = cv2.VideoCapture("/Users/mlykotom/Downloads/ICNet_" + video_postfix + "_drive.avi")
pspnet_gt = cv2.VideoCapture("/Users/mlykotom/Downloads/pspnet_stuttgart_02.avi")

while True:
    ret, frame = video.read()
    ret_warp, frame_warp = warp_video.read()
    ret_norm, frame_norm = norm_video.read()
    ret_psp, frame_psp = pspnet_gt.read()

    if not (ret or ret_warp or ret_norm):
        warp_video.release()
        norm_video.release()
        print("!! Released Video Resource")
        break

    frame_norm_gray = cv2.cvtColor(frame_norm, cv2.COLOR_RGB2GRAY)
    frame_warp_gray = cv2.cvtColor(frame_warp, cv2.COLOR_RGB2GRAY)

    # cv2.imshow("warp", frame_warp)
    # cv2.imshow("norm", frame_norm)
    # cv2.imshow("diff_n_w", cv2.absdiff(frame_norm_gray, frame_warp_gray))

    # frame_small = cv2.resize(frame, (512, 256))
    diff = cv2.cvtColor(cv2.absdiff(frame_norm_gray, frame_warp_gray), cv2.COLOR_GRAY2BGR)
    gt_frame = np.concatenate((frame_psp, diff), axis=1)
    our_frame = np.concatenate((frame_norm, frame_warp), axis=1)

    whole_frame = np.concatenate((gt_frame, our_frame), axis=0)
    cv2.imshow("normal|diff\icnet|warp", whole_frame)
    key = cv2.waitKey()
    if key == 32:
        print("-- video paused")
        while True:
            key = cv2.waitKey(1 )
            if key == 32:
                break
        print("-- video unpaused")



    # all_videos = np.hstack(frame_norm)
