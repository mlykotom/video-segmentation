import os

from keras import Sequential

from models.flow_cnn import FlowCNN

if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    hmm = FlowCNN((288, 480, 3), 30)
    print(hmm.summary())
    hmm.plot_model()
