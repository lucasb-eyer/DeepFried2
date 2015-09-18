import numpy as _np


def eye(gain=1):
    def init(shape, fan):
        return gain * _np.eye(*shape)
    return init
