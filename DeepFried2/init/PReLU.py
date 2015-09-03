import DeepFried2 as df
import numpy as _np

def prelu(gain=1):
    return df.init.xavier(gain * _np.sqrt(2))

def preluN(gain=1):
    return df.init.xavierN(gain * _np.sqrt(2))
