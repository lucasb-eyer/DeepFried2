import numpy as _np


def normal(std):
    def init(shape, fan):
        return std*_np.random.randn(*shape)
    return init
