from DeepFried2.zoo.download import download as _download

import numpy as _np
import scipy.io as _sio


def data():
    fname_train = _download('http://ufldl.stanford.edu/housenumbers/train_32x32.mat')
    fname_test = _download('http://ufldl.stanford.edu/housenumbers/test_32x32.mat')
    fname_extra = _download('http://ufldl.stanford.edu/housenumbers/extra_32x32.mat')

    def loadxy(fname):
        mat = _sio.loadmat(fname)
        X = mat['X'].transpose(3, 2, 0, 1).astype(_np.float32)
        X /= 255
        y = mat['y'][:,0].astype(_np.int64)  # For convenience/avoid stupid bugs.
        return X, y

    return loadxy(fname_train), loadxy(fname_test), loadxy(fname_extra)

