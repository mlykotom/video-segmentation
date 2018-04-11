# model

import config
from generator import *
from models import *

batch_size = 1
target_size = 256, 512
# target_size = (1052, 1914)
dataset_path = config.data_path()

model = 'segnet_warp_diff'
print("-- model %s" % model)

if model == 'segnet_warp_diff':
    datagen = CityscapesFlowGenerator(config.data_path(), flow_with_diff=True)
    model = SegNetWarpDiff(config.target_size(), datagen.n_classes)
    # model.k.load_weights('../../weights/city/SegNetWarpDiff/warp_diff.h5')
    model.k.load_weights('/home/mlyko/weights/city/SegNetWarpDiff/diff_p0_w123_s04_aug.h5')
    # model.k.load_weights('/home/mlyko/weights/city/SegNetWarpDiff/diff_p0_w123_s04_2.h5')
    eval_batch_size = 5

elif model == 'icnet':
    raise NotImplementedError("Unknown model")
    # eval_batch_size = 5
else:
    raise NotImplementedError("Unknown model")

model.compile()
# model.summary()


prediction = model.k.evaluate_generator(
    generator=datagen.flow('val', eval_batch_size, target_size),
    steps=datagen.steps_per_epoch('val', eval_batch_size),
)

print(model.k.metrics_names)
print(prediction)
