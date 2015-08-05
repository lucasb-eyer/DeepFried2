import numpy as _np


def uniform(low, high):
    def init(shape, fan):
        return _np.random.uniform(low=low, high=high, size=shape)
    return init
