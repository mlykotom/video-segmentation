from abc import ABCMeta


class BaseExperiment:
    __metaclass__ = ABCMeta

    def prepare_model(self):
        pass

    def prepare_callbacks(self):
        pass

    def prepare_data(self):
        pass

    def run(self):
        pass


class MobileNetGTAExperiment:
    pass
