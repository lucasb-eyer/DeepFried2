import numpy as _np
from DeepFried2.init import xavier, xavierN

def prelu(gain=1):
    return xavier(gain * _np.sqrt(2))

def preluN(gain=1):
    return xavierN(gain * _np.sqrt(2))
