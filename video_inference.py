import argparse
import os

import cv2
import numpy as np

import config
from generator import CityscapesFlowGenerator
from models import *


class VideoEvaluator:
    @staticmethod
    def get_gpu_name():
        from tensorflow.python.client import device_lib

        local_device_protos = device_lib.list_local_devices()
        l = [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']
        s = ''
        for t in l:
            s += t[t.find("name: ") + len("name: "):t.find(", pci")] + " "
        return s

    def select_device(self, gid=None):
        if gid is not None:
            if gid == "cpu":
                # use CPU
                print("-- Using CPU")
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                gpu_name = "CPU"
            else:
                print("-- Using GPU id %s" % gid)
                os.environ["CUDA_VISIBLE_DEVICES"] = gid
                gpu_name = self.get_gpu_name()
        else:
            gpu_name = self.get_gpu_name()

        return gpu_name

    def _open_video(self, input_file):
        print("-- reading file %s" % input_file)
        if not os.path.isfile(input_file):
            raise Exception('File %s not found' % input_file)

        vid = cv2.VideoCapture(input_file)
        frames_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vid.get(cv2.CAP_PROP_FPS)
        print("-- frames %d" % frames_count, "fps %d" % fps)
        return vid, fps

    def _prepare_output(self, file_as_input, model, fps=30):
        """
        :param file_as_input:
        :param BaseModel model:
        :param fps:
        :return:
        """
        out_file_prefix = os.path.split(os.path.splitext(file_as_input)[0])[-1]
        output_file = out_file_prefix + '_' + model.name + '.avi'  # e.g. stuttgart_00_ICNetWarp0.avi

        print("-- preparing output to file %s" % output_file)
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        return cv2.VideoWriter(output_file, fourcc, float(fps), (model.target_size[1], model.target_size[0]))

    _last_frame = None
    _last_prediction = None
    _models = []

    def process_frame(self, frame, model, verbose=1):
        """

        :param frame:
        :param BaseModel model:
        :return list: should be a list with the prediction (because of compatibility with warping prediciton)
        """

        frame = cv2.resize(frame, config.target_size()[::-1])
        frame_norm = datagen.normalize(frame, config.target_size())

        input = [np.array([frame_norm])]
        return [model.k.predict(input, 1, verbose)]

    def process_frame_warping(self, frame, last_frame, model, last_prediction=None, verbose=1):
        """

        :param frame:
        :param last_frame:
        :param BaseModel model:
        :param verbose:
        :return:
        """

        flow = datagen.calc_optical_flow(frame, last_frame)
        frame_norm = datagen.normalize(frame, config.target_size())
        last_frame_norm = datagen.normalize(last_frame, config.target_size())

        input_with_flow = [
            np.array([last_frame_norm]),
            np.array([frame_norm]),
            np.array([flow])
        ]

        if last_prediction is not None:
            input_with_flow += last_prediction[1:]
        else:
            # takes all layers to warp (should be specified by model) and creates array of ones of the same shape
            input_with_flow += [np.ones((1,) + out_shape[1:]) for out_shape in model.k.output_shape[1:]]

        all_predictions = model.k.predict(input_with_flow, 1, verbose)
        return all_predictions

    def process_video(self, datagen, input_file, skip_from_start=0, until_frame=None):
        # may process more models
        for model_params in self._models:
            # load input video
            vid, fps = self._open_video(input_file)
            try:
                # load model
                model = model_params['model']
                model.k.load_weights(model_params['weights'], by_name=True)
                model.compile()

                print('-- predicting model %s' % model.name)

                # prepare output file for the model and input file
                out = self._prepare_output(input_file, model, fps)
                # reset old state
                self._last_frame = None
                self._last_prediction = None
                frame_i = 0

                while True:
                    ret, frame = vid.read()
                    if not ret:
                        vid.release()
                        print("-- Released Video Resource")
                        break

                    # skipping boring part of video
                    if frame_i < skip_from_start:
                        frame_i += 1
                        continue

                    # cut from end
                    if until_frame is not None and frame_i > until_frame:
                        print(" -- Finishing at frame %d" % until_frame)
                        break

                    frame = cv2.resize(frame, config.target_size()[::-1])

                    if model_params['warp']:
                        if self._last_frame is None:
                            self._last_frame = frame

                        predictions = self.process_frame_warping(frame, self._last_frame, model, self._last_prediction)
                    else:
                        predictions = self.process_frame(frame, model)

                    colored_prediction = datagen.one_hot_to_bgr(predictions[0], config.target_size(), datagen.n_classes, datagen.labels)
                    out.write(colored_prediction)

                    print('-- processed frame %d' % frame_i)
                    self._last_frame = frame
                    self._last_prediction = predictions
                    frame_i += 1
                out.release()
                vid.release()

                print("-- Finished input stream")
            except KeyboardInterrupt:
                # Release the Video Device
                vid.release()
                # Message to be displayed after releasing the device
                print("-- Released Video Resource")

    def load_model(self, params):
        self._models.append(params)


if __name__ == '__main__':
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Video evaluation')

        parser.add_argument('--gid',
                            help='GPU id',
                            default=None)

        parser.add_argument(
            '-i', '--input',
            help='Input file',
            default='/home/mlyko/data/stuttgart_00.mp4'
        )

        parser.add_argument(
            '-o', '--output',
            help='Output file',
            default=None
        )

        args = parser.parse_args()
        return args


    args = parse_arguments()

    size = config.target_size()

    videoEvaluator = VideoEvaluator()
    videoEvaluator.select_device(args.gid)

    datagen = CityscapesFlowGenerator(config.data_path())

    videoEvaluator.load_model({
        'model': ICNet(config.target_size(), datagen.n_classes, for_training=False),
        'weights': config.weights_path() + 'city/rel/ICNet/1612:37e200.b8.lr-0.001000._dec-0.000000.of-farn.h5',
        'warp': False
    })

    videoEvaluator.load_model({
        'model': ICNetWarp0(config.target_size(), datagen.n_classes, for_training=False),
        'weights': config.weights_path() + 'city/rel/ICNetWarp0/fin.e150.b8.lr-0.005000._dec-0.000000.of-farn.h5',
        'warp': True
    })

    videoEvaluator.process_video(datagen, args.input)
