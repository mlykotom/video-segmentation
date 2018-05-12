import argparse
import cv2
import os


class VideoEvaluation:
    pass

    def open_video(self, input_file):
        print("-- reading input %s" % input_file)
        vid = cv2.VideoCapture("/home/mlyko/data/%s" % input_file)
        frames_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vid.get(cv2.CAP_PROP_FPS)
        print("frames %d" % frames_count, "fps %d" % fps)
        return vid, fps

    @staticmethod
    def getGPUname():
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
                gpu_name = self.getGPUname()
        else:
            gpu_name = self.getGPUname()

        return gpu_name

    def prepare(self, model_name):
        pass

    def prep_out_video(self, out_file, fps, target_size):
        print("-- preparing output to file %s" % out_file)
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        return cv2.VideoWriter(out_file, fourcc, float(fps), target_size[:-1])


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

    videoEvaluator = VideoEvaluation()
    videoEvaluator.select_device(args.gid)
    videoEvaluator.open_video(args.input)

    videoEvaluator.prepare(args.model)

    # TODO sequentially load the same portion of video for different models!

    # TODO finish