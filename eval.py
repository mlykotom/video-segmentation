import config
from generator import *
from models import *

target_size = 256, 512
dataset_path = config.data_path()
weights_path = config.weights_path() + 'city/rel/'

# %%

# eval_batch_size = 4

# model = 'segnet_warp_diff'
# print("-- model %s" % model)
#
# # if model == 'segnet_warp_diff':
# #     datagen = CityscapesFlowGenerator(config.data_path())
#     model = SegNetWarp(config.target_size(), datagen.n_classes)
#     # model.k.load_weights('../../weights/city/SegNetWarpDiff/warp_diff.h5')
#     model.k.load_weights('/home/mlyko/weights/city/SegNetWarpDiff/diff_p0_w123_s04_aug.h5')
#     # model.k.load_weights('/home/mlyko/weights/city/SegNetWarpDiff/diff_p0_w123_s04_2.h5')
#     eval_batch_size = 5
# #
# # elif model == 'icnet':
# #     raise NotImplementedError("Unknown model")
# #     # eval_batch_size = 5
# # else:
# #     raise NotImplementedError("Unknown model")
#
#
# if model == 'icnet':


# # %%
# eval_batch_size = 8
#
# datagen = CityscapesGeneratorForICNet(config.data_path())
# model = ICNet(config.target_size(), datagen.n_classes)
# model.k.load_weights(weights_path + 'ICNet/baseline.e150.b8.lr=0.001000._dec=0.051000.of=farn.h5')
#
# model.compile()
# # model.summary()
#
#
# prediction = model.k.evaluate_generator(
#     generator=datagen.flow('val', eval_batch_size, target_size),
#     steps=datagen.steps_per_epoch('val', eval_batch_size),
# )
#
# result = dict(zip(model.k.metrics_names, prediction))
# print(result)


# %%
eval_batch_size = 8

datagen = CityscapesFlowGeneratorForICNet(config.data_path())
model = ICNetWarp1(config.target_size(), datagen.n_classes)
model.k.load_weights(weights_path + 'ICNetWarp1/3019:40e150.b8.lr=0.001000._dec=0.051000.of=farn.h5', by_name=True)

model.compile()
# model.summary()


prediction = model.k.evaluate_generator(
    generator=datagen.flow('val', eval_batch_size, target_size),
    steps=datagen.steps_per_epoch('val', eval_batch_size),
)

result = dict(zip(model.k.metrics_names, prediction))
print(result)

