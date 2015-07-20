from DeepFried2.zoo.download import download as _download

import numpy as _np
from tarfile import open as _taropen
try:  # Py2 compatibility
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def data():
    fname = _download('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    with _taropen(fname, 'r') as f:
        # The first four batches are used as training set...
        datas, labels = [], []
        for i in range(1, 5):
            with f.extractfile('cifar-10-batches-py/data_batch_' + str(i)) as b:
                batch = _pickle.load(b, encoding='latin1')
                datas.append(_np.array(batch['data'], dtype=_np.float32))
                labels.append(_np.array(batch['labels']))
        Xtr = _np.concatenate(datas)
        ytr = _np.concatenate(labels)
        Xtr /= 255

        # ... and the fifth as validation set as described in cuda-convnet:
        # https://code.google.com/p/cuda-convnet/wiki/Methodology
        with f.extractfile('cifar-10-batches-py/data_batch_5') as b:
            batch = _pickle.load(b, encoding='latin1')
        Xva = _np.array(batch['data'], dtype=_np.float32)
        yva = _np.array(batch['labels'])
        Xva /= 255

        with f.extractfile('cifar-10-batches-py/test_batch') as b:
            batch = _pickle.load(b, encoding='latin1')
        Xte = _np.array(batch['data'], dtype=_np.float32)
        yte = _np.array(batch['labels'])
        Xte /= 255

    return (Xtr, ytr), (Xva, yva), (Xte, yte)

