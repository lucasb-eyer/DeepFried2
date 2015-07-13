import numpy as _np


def const(value):
    def init(shape, fan):
        return _np.full(shape, value)
    return init
