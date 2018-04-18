import argparse
import datetime
import os

import cv2
import numpy as np

import config
from generator import *
from models import *


def getGPUname():
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    l = [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']
    s = ''
    for t in l:
        s += t[t.find("name: ") + len("name: "):t.find(", pci")] + " "
    return s


font_thickness = 2
font_scale = 0.4


def _put_text(out, text, where, color=(255, 255, 255)):
    cv2.putText(out, text, where, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    cv2.putText(out, text, where, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)


def write_text(alpha_blended, model, diff, counter, time_sum, gpu_name, gpu_model_info=(100, 50),
               prediction_info=(100, 70)):
    _put_text(alpha_blended, '%s %s' % (gpu_name, model), gpu_model_info)

    pred_time = diff.microseconds / 1000.0
    pred_fps = 1000000.0 / diff.microseconds
    avg_time = time_sum / (counter + 1)
    avg_fps = 1000.0 / (time_sum / (counter + 1))

    _put_text(alpha_blended, 'Prediction time: %.0fms (%.1f fps) AVG: %.0fms (%.1f fps)' %
              (pred_time, pred_fps, avg_time, avg_fps),
              gpu_model_info)

    print('Prediction time: %.0fms (%.1f fps) AVG: %.0fms (%.1f fps)' % (pred_time, pred_fps, avg_time, avg_fps))


def open_video(input_file):
    print("-- reading input %s" % input_file)
    vid = cv2.VideoCapture("/home/mlyko/data/%s" % input_file)
    frames_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    print("frames %d" % frames_count, "fps %d" % fps)
    return vid, fps


def prep_out_video(out_file, fps, height, width):
    print("-- preparing output to file %s" % out_file)
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    return cv2.VideoWriter(out_file, fourcc, float(fps), (width, height))


def predict_frame(input, model, datagen, old_labels=False):
    start = datetime.datetime.now()
    prediction = model.k.predict(input, 1, verbose=1)
    end = datetime.datetime.now()
    diff = end - start

    colored_class_image = datagen.one_hot_to_bgr(prediction, config.target_size(), datagen.n_classes,
                                                 datagen.old_labels if old_labels else datagen.labels)
    return colored_class_image, diff


if __name__ == '__main__':
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Train model in keras')

        parser.add_argument('-m', '--model',
                            help='Model to train [segnet, mobile_unet]',
                            default='segnet_warp')

        parser.add_argument('--gid',
                            help='GPU id',
                            default=None)

        parser.add_argument(
            '-i', '--input',
            help='Input file (takes from /home/mlyko/data/)(drive.mp4)',
            default='drive.mp4'
        )

        parser.add_argument(
            '-o', '--output',
            help='Output file'
        )

        args = parser.parse_args()
        return args


    args = parse_arguments()
    vid, fps = open_video(args.input)

    if args.gid is not None:
        if args.gid == "cpu":
            # use CPU
            print("-- Using CPU")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            gpu_name = "CPU"
        else:
            print("-- Using GPU id %s" % args.gid)
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gid
            gpu_name = getGPUname()
    else:
        gpu_name = getGPUname()

    model_name = args.model

    # if model_name == 'segnet':
    #     raise NotImplementedError("Model not specified")
    #     # datagen = CityscapesFlowGenerator(config.data_path())
    #     # model = SegNetWarpDiff(config.target_size(), datagen.n_classes)
    # else:
    #     model_name = 'segnet_warp'
    #     datagen = CityscapesFlowGenerator(config.data_path())
    #     model = SegNetWarpDiff123(config.target_size(), datagen.n_classes)
    #     # model.k.load_weights('../../weights/city/SegNetWarpDiff/warp_diff.h5')
    #     # model.k.load_weights('/home/mlyko/weights/city/SegNetWarpDiff/diff_p0_w123_s04_aug.h5')
    #     # model.k.load_weights('/home/mlyko/weights/city/rel/SegNetWarpDiff/diff_p0_w3_s04.h5')
    #     model.k.load_weights(
    #         '/home/mlyko/weights/city/rel/SegNetWarpDiff123/diff_p0_w123_s04_norm_eq_hist_dec_0001_g2.h5')
    #     # model.k.load_weights('/home/mlyko/weights/city/SegNetWarpDiff/diff_p0_w123_s04_2.h5')

    datagen = CityscapesFlowGenerator(config.data_path())
    # model_left = SegNet(config.target_size(), datagen.n_classes)
    # model_left.k.load_weights(config.weights_path() + 'city/rel/SegNet/normal_eq_hist.h5')
    # model_left.compile()

    model_left = MobileUNetWarp2(config.target_size(), datagen.n_classes)
    # model_left.k.load_weights(config.weights_path() + 'city/rel/MobileUNetWarpmob_unet_b6_warp1_spat0.2.h5')
    model_left.k.load_weights(config.weights_path() + 'city/rel/MobileUNetWarp2/omg_po_ranu.h5')
    model_left.compile()

    model_right = SegNetWarpDiff123(config.target_size(), datagen.n_classes)
    model_right.k.load_weights(config.weights_path() + 'city/rel/SegNetWarpDiff123/diff_p0_w123_s03_b4_dec0001_g2.h5')
    model_right.compile()

    print("-- models loaded %s|%s" % (model_left.name, model_right.name))

    height = config.target_size()[0]
    width = config.target_size()[1] * 2
    out = prep_out_video(args.output, fps, height, width)
    print("-- reading data")
    try:
        time_sum = 0
        i = 0
        last_frame = None
        while True:
            ret, frame = vid.read()
            if not ret:
                vid.release()
                print("!! Released Video Resource")
                break

            # skipping boring part of video
            if i < 4000:
                i += 1
                continue

            frame = cv2.resize(frame, config.target_size()[::-1])
            if last_frame is None:
                last_frame = frame

            flow = datagen.calc_optical_flow(last_frame, frame)

            # TODO                 self.optical_flow.setUseSpatialPropagation(True)

            frame_norm = datagen.normalize(frame, config.target_size())
            last_frame_norm = datagen.normalize(last_frame, config.target_size())

            input = [
                np.array([last_frame_norm]),
                np.array([frame_norm]),
                np.array([flow]),
                np.array([frame_norm - last_frame_norm])
            ]

            # np.array([frame_norm])
            colored_left, diff_left = predict_frame(input, model_left, datagen)
            colored_right, diff_right = predict_frame(input, model_right, datagen)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            alpha_blended_both = (0.6 * colored_left + 0.4 * img).astype('uint8'), \
                                 (0.6 * colored_right + 0.4 * img).astype('uint8')

            both_blended = np.hstack(alpha_blended_both)

            time_sum += diff_left.microseconds / 1000.0
            print(i, diff_left.microseconds / 1000.0, 'ms')
            # write_text(alpha_blended, model_name, diff_left, i, time_sum, gpu_name)

            out.write(both_blended)
            last_frame = frame

            i += 1

    except KeyboardInterrupt:
        # Release the Video Device
        vid.release()
        # Message to be displayed after releasing the device
        print("Released Video Resource")
