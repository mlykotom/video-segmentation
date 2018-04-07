import argparse
import os
from time import gmtime, strftime

import config
from trainer import Trainer


def train(dataset_path, model_name='mobile_unet', run_name='', debug_samples=0, restart_training=False,
          batch_size=None, n_gpu=1, summaries=False, epochs=150, early_stopping=10):
    target_size = config.target_size()
    batch_size = batch_size or 2

    trainer = Trainer(model_name, dataset_path, target_size, batch_size, n_gpu, debug_samples, early_stopping)
    trainer.model.compile()

    if summaries:
        trainer.summaries()

    # train model
    trainer.fit_model(
        run_name=run_name,
        epochs=epochs,
        restart_training=restart_training
    )


if __name__ == '__main__':
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Train model in keras')
        parser.add_argument('-r', '--restart',
                            action='store_true',
                            help='Restarts training from last saved epoch',
                            default=False)

        parser.add_argument('-d', '--debug',
                            help='Just debug training (number to pick from dataset)',
                            default=0)

        parser.add_argument('--summaries',
                            action='store_true',
                            help='If should plot model and summary',
                            default=False)

        parser.add_argument('-g', '--gpus',
                            help='Number of GPUs used for training',
                            default=1)

        parser.add_argument('-m', '--model',
                            help='Model to train [segnet, mobile_unet]',
                            default='segnet')

        parser.add_argument('-b', '--batch',
                            help='Batch size',
                            default=2)

        parser.add_argument('-e', '--epochs',
                            help='Number of epochs',
                            default=150)

        parser.add_argument('-s', '--stop',
                            help='Early stopping',
                            default=5)

        parser.add_argument('--gid',
                            help='GPU id',
                            default=None)

        parser.add_argument(
            '-n', '--name',
            help='Run Name',
            default=strftime("%Y_%m_%d_%H:%M", gmtime())
        )

        args = parser.parse_args()
        return args


    args = parse_arguments()

    if args.gpus > 1 and args.gid is not None:
        raise Exception("Can't be multi model and gpu specified")

    dataset_path = config.data_path()

    print("---------------")
    print('dataset path', dataset_path)
    print("GPUs number", args.gpus)
    print("selected GPU ID", args.gid)
    print("---------------")
    print('model', args.model)
    print("---------------")
    print("debug samples", args.debug)
    print("restart training", args.restart)
    print("---------------")
    print("batch size", args.batch)
    print("run name", args.name)
    print("---------------")

    if args.gid is not None:
        if args.gid == "cpu":
            # use CPU
            print("-- Using CPU")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            print("-- Using GPU id %s" % args.gid)
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gid

    try:
        train(
            dataset_path=dataset_path,
            model_name=args.model,
            run_name=args.name,
            debug_samples=int(args.debug),
            restart_training=args.restart,
            batch_size=int(args.batch),
            n_gpu=int(args.gpus),
            summaries=args.summaries,
            epochs=int(args.epochs),
            early_stopping=int(args.stop)
        )
    except KeyboardInterrupt:
        print("Keyboard interrupted")
