import numpy as np

from generator import *
from generator import cityscapes_labels
from models import *

target_size = 256, 512

import config


def layer_to_visualize(img_to_visualize, model, layer):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])

    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + X)

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)
    print ('Shape of conv:', convolutions.shape)

    return convolutions


def get_layer_output(input, model, layer, index=None):
    all_inputs = model.inputs + [K.learning_phase()]
    all_outputs = [layer.output if index is None else layer.get_output_at(index)]
    output = K.function(all_inputs, all_outputs)

    convolutions = output(input + [0])
    convolutions = np.squeeze(convolutions)
    print ('Shape of conv:', convolutions.shape)

    return convolutions


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    labels = cityscapes_labels.labels
    n_classes = len(labels)
    batch_size = 2
    epochs = 10

    dataset_path = config.data_path()

    # segnet
    # model = MobileUNetWarp(target_size, n_classes)
    # model = ICNetWarp1(target_size, n_classes)
    # model = SegnetWarp2(target_size, n_classes, for_training=False)
    model = ICNetWarp2(target_size, n_classes, for_training=False)
    model.compile()
    print(model.summary())

    # datagen = CityscapesFlowGenerator(dataset_path, debug_samples=20, prev_skip=0)
    datagen = CityscapesFlowGeneratorForICNet(dataset_path, debug_samples=20, prev_skip=0)

    # TODO weights not saved with _2 or _3 postfix!!!!
    # model.k.load_weights('/home/mlyko/weights/city/rel/ICNetWarp2/hopee200.b8.lr=0.001000._dec=0.055000.h5', by_name=True)

    x = 0

    for imgBatch, labelBatch in datagen.flow('train', 1, target_size):
        if x < 5:
            x += 1
            continue

        imgOld = imgBatch[0][0]
        imgNew = imgBatch[1][0]
        inpFlow = imgBatch[2][0]

        print("shapes", imgOld.shape, imgNew.shape, inpFlow.shape)

        model.k.load_weights('/home/mlyko/weights/city/deb/ICNetWarp2/8/no_final_wrfe200.b8.lr=0.001000._dec=0.055000.h5', by_name=True)

        # input_data = [
        #     np.array([imgOld]),
        #     np.array([imgNew]),
        #     np.array([inpFlow]),
        #     np.zeros((1, 32, 64, 256)),
        # ]
        #
        # finish = Model(
        #     inputs=model.k.inputs,
        #     outputs=[
        #         model.k.get_layer('conv_block_3').get_output_at(1),
        #         model.k.get_layer('conv2d_12').output
        #     ]
        # )
        #
        # prediction = finish.predict(input_data, 1, 1)
        # print(prediction[0].shape, prediction[1].shape)
        #
        # break
        #
        # imgBatch = input_data

        # img_diff = layer_to_visualize(imgBatch, model.k, model.k.get_layer('img_diff'), layers)
        # transformed_flow = get_layer_output(imgBatch, model.k, model.k.get_layer('transformed_flow'))
        transformed_flow_resize = get_layer_output(imgBatch, model.k, model.k.get_layer('resize_bilinear_1'))
        img_diff = get_layer_output(imgBatch, model.k, model.k.get_layer('img_diff'))
        concat_1 = get_layer_output(imgBatch, model.k, model.k.get_layer('concatenate_1'))
        conv_1 = get_layer_output(imgBatch, model.k, model.k.get_layer('conv2d_1'))
        conv_2 = get_layer_output(imgBatch, model.k, model.k.get_layer('conv2d_2'))
        conv_3 = get_layer_output(imgBatch, model.k, model.k.get_layer('conv2d_3'))
        # concat_2 = get_layer_output(imgBatch, model.k, model.k.get_layer('concatenate_2'))

        # print(inpFlow[0][0:10])
        # print(inpFlow[1][0:10])
        # print("--")
        # fl = datagen.flow_to_bgr(inpFlow, target_size)
        # print("Angl", fl[0][0:10])
        # print("m", fl[1][0:10])
        # print("------")
        # print(transformed_flow[0][0:10])
        # print(transformed_flow[1][0:10])
        # trfl = datagen.flow_to_bgr(transformed_flow, target_size)
        # print("Angl", trfl[0][0:10])
        # print("m", trfl[1][0:10])
        # # break

        transformed_flow = conv_3
        # break
        cv2.imshow("flow_GT", datagen.flow_to_bgr(inpFlow, target_size))
        cv2.imshow("flow3x3_open", datagen.flow_to_bgr(transformed_flow, target_size))
        # cv2.imshow("flow_small", datagen.flow_to_bgr(transformed_flow_resize, transformed_flow_resize.shape[:-1]))
        # cv2.imshow("img", datagen.denormalize(imgNew))

        # print("diff shape", img_diff.shape)
        # cv2.imshow("img_diff", img_diff)
        cv2.waitKey()
