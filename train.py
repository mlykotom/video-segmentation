import argparse
import os
from time import gmtime, strftime

import config
from trainer import Trainer

if __name__ == '__main__':
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Train model in keras')
        parser.add_argument(
            '-r', '--restart',
            action='store_true',
            help='Restarts training from last saved epoch',
            default=False
        )

        parser.add_argument(
            '-d', '--debug',
            help='Just debug training (number to pick from dataset)',
            default=0
        )

        parser.add_argument(
            '--summaries',
            action='store_true',
            help='If should plot model and summary',
            default=False)

        parser.add_argument(
            '-g', '--gpus',
            help='Number of GPUs used for training',
            default=1
        )

        parser.add_argument(
            '-m', '--model',
            help='Model to train [segnet, mobile_unet]',
            default='segnet'
        )

        parser.add_argument(
            '-b', '--batch',
            help='Batch size',
            default=2
        )

        parser.add_argument(
            '-e', '--epochs',
            help='Number of epochs',
            default=200
        )

        parser.add_argument(
            '-s', '--stop',
            help='Early stopping',
            default=20
        )

        parser.add_argument(
            '--gid',
            help='GPU id',
            default=None
        )

        parser.add_argument(
            '-lr',
            help='Learning rate',
            default=None
        )

        parser.add_argument(
            '--dec',
            help='Learning rate decay',
            default=None
        )

        parser.add_argument(
            '-n', '--name',
            help='Run Name',
            default=strftime("%d%H:%M", gmtime())
        )

        parser.add_argument(
            '-o', '--optic',
            help='Optical flow',
            default='farn'
        )

        parser.add_argument(
            '--height',
            help='Target image height',
            default=config.target_size()[0]
        )

        parser.add_argument(
            '--width',
            help='Target image width',
            default=config.target_size()[1]
        )

        parser.add_argument(
            '--aug',
            help='Data Augmentation',
            default=False
        )

        parser.add_argument(
            '--workers',
            help='Workers',
            default=1
        )
        parser.add_argument(
            '--multiprocess',
            help='Multiprocess',
            default=False
        )

        parser.add_argument(
            '--queue',
            help='Max queue',
            default=50
        )

        parser.add_argument(
            '--gpu_percent',
            help='How much GPU memory will be taken',
            default=None
        )

        args = parser.parse_args()
        return args


    args = parse_arguments()

    if args.gpus > 1 and args.gid is not None:
        raise Exception("Can't be multi model and gpu specified")

    dataset_path = config.data_path()

    multiprocess = args.multiprocess is not None and (args.multiprocess == 'true' or args.multiprocess == 'True')

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
    print("data augmentation", args.aug)
    print("---------------")
    print("workers", args.workers, "multiprocess", multiprocess)
    print("max_queue", args.queue)
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

    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf

    if args.gpu_percent is not None:
        print("--Using %f gpu" % float(args.gpu_percent))
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = float(args.gpu_percent)
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    try:
        epochs = int(args.epochs)
        target_size = int(args.height), int(args.width)
        batch_size = int(args.batch) or 2
        debug_samples = int(args.debug)
        early_stopping = int(args.stop)
        summaries = args.summaries
        n_gpu = int(args.gpus)

        restart_training = args.restart
        run_name = args.name

        optical_flow_type = args.optic

        print("target size", target_size)
        print("---------------")

        data_augmentation = bool(args.aug)

        trainer = Trainer(
            model_name=args.model,
            dataset_path=dataset_path,
            target_size=target_size,
            batch_size=batch_size,
            n_gpu=n_gpu,
            debug_samples=debug_samples,
            early_stopping=early_stopping,
            optical_flow_type=optical_flow_type,
            data_augmentation=data_augmentation
        )

        trainer.model.compile(
            lr=float(args.lr) if args.lr is not None else None,
            lr_decay=float(args.dec) if args.dec is not None else 0.
        )

        if summaries:
            trainer.summaries()

        # train model
        trainer.fit_model(
            run_name=run_name,
            epochs=epochs,
            restart_training=restart_training,
            workers=int(args.workers),
            max_queue=int(args.queue),
            multiprocess=multiprocess
        )
    except KeyboardInterrupt:
        print("Keyboard interrupted")
