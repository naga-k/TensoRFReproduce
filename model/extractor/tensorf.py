from model.extractor.tensorf_base import TensorBase


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)
        pass

    def init_one_svd(self, n_component, gridSize, scale, device):
        