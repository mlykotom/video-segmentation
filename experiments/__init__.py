from abc import ABCMeta, abstractproperty

from ..generator import CamVidGenerator, CamVidFlowGenerator


# TODO include in trainer

class Experiment:
    __metaclass__ = ABCMeta

    @abstractproperty
    def target_size(self):
        pass

    @abstractproperty
    def generator(self):
        pass

    def __init__(self, dataset_path, debug_samples):
        self.dataset_path = dataset_path
        self.debug_samples = debug_samples


class CamVidExperiment(Experiment):
    __metaclass__ = ABCMeta

    def __init__(self, dataset_path, debug_samples, use_flow=False):
        self.use_flow = use_flow
        super(CamVidExperiment, self).__init__(dataset_path, debug_samples)

    @property
    def target_size(self):
        return 288, 480

    @property
    def generator(self):
        if self.use_flow:
            return CamVidFlowGenerator(self.dataset_path, debug_samples=self.debug_samples)
        else:
            return CamVidGenerator(self.dataset_path, debug_samples=self.debug_samples)


class CityscapesExperiment(Experiment):
    __metaclass__ = ABCMeta

    @property
    def target_size(self):
        return 256, 512
