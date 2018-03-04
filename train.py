import argparse
from time import gmtime, strftime

import config
from callbacks import *
from trainer import Trainer


def train(dataset_path, model_name='mobile_unet', run_name='', is_debug=False, restart_training=False,
          batch_size=None, n_gpu=1, summaries=False):
    # TODO make smaller
    # target_size = 360, 648
    # target_size = 384, 640
    target_size = 288, 480
    # target_size = (1052, 1914) # original

    batch_size = batch_size or 2
    epochs = 200

    trainer = Trainer(model_name, dataset_path, target_size, batch_size, n_gpu, is_debug)
    model = trainer.compile_model()

    if summaries:
        trainer.summaries()

    # train model
    trainer.fit_model(
        run_name=run_name,
        epochs=epochs,
        restart_training=restart_training
    )

    # save final model
    model.save_final(run_name, epochs)


if __name__ == '__main__':
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Train model in keras')
        parser.add_argument('-r', '--restart',
                            action='store_true',
                            help='Restarts training from last saved epoch',
                            default=False)

        parser.add_argument('-d', '--debug',
                            action='store_true',
                            help='Just debug training (few images from dataset)',
                            default=False)

        parser.add_argument('--summaries',
                            action='store_true',
                            help='If should plot model and summary',
                            default=False)

        parser.add_argument('-g', '--gpus',
                            help='Number of GPUs used for training',
                            default=1)

        parser.add_argument('-m', '--model',
                            help='Model to train [segnet, mobile_unet]',
                            default='mobile_unet')

        parser.add_argument('-b', '--batch',
                            help='Batch size',
                            default=2)

        parser.add_argument('--gid',
                            help='GPU id',
                            default=None)

        args = parser.parse_args()
        return args


    args = parse_arguments()

    if args.gpus > 1 and args.gid is not None:
        raise Exception("Can't be multimodel and gpu specified")

    dataset_path = config.data_path()

    print("---------------")
    print('dataset path', dataset_path)
    print("GPUs number", args.gpus)
    print("selected GPU ID", args.gid)
    print("---------------")
    print('model', args.model)
    print("---------------")
    print("is debug", args.debug)
    print("restart training", args.restart)
    print("---------------")
    print("batch size", args.batch)
    print("---------------")

    if args.gid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gid

    run_name = strftime("%Y_%m_%d_%H:%M", gmtime())

    try:
        train(dataset_path, args.model, run_name,
              is_debug=args.debug,
              restart_training=args.restart,
              batch_size=int(args.batch),
              n_gpu=int(args.gpus),
              summaries=args.summaries)
    except KeyboardInterrupt:
        print("Keyboard interrupted")
