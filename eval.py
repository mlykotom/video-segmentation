# model

import config
from generator import *
from models import *

batch_size = 1
target_size = 256, 512
# target_size = (1052, 1914)
dataset_path = config.data_path()

# model = MobileUNet(target_size, n_classes)
# model.k.load_weights('weights/MobileUNet_2018_02_15_09:34_cat_acc-0.89.hdf5')
datagen = CityscapesFlowGenerator(dataset_path, flow_with_diff=True)

model = SegNetWarpDiff(target_size, datagen.n_classes)
model.k.load_weights('../../weights/city/SegNetWarpDiff/warp_diff.h5')
model.compile()
# model.summary()

eval_batch_size = 5

prediction = model.k.evaluate_generator(
    generator=datagen.flow('test', eval_batch_size, target_size),
    steps=datagen.steps_per_epoch('test', eval_batch_size),
)

print(model.k.metrics_names)
print(prediction)
