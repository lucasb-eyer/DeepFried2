from DeepFried2 import floatX as _fX
from DeepFried2.zoo.download import download as _download

import scipy.io as _sio


def data():
    fname = _download('http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat')
    mat = _sio.loadmat(fname)

    X = mat['ff'].T.reshape(-1, 28, 20).astype(_fX)
    X /= 255
    return X
