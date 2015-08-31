from DeepFried2.zoo.download import download as _download

import sys as _sys
import gzip as _gzip
try:  # Py2 compatibility
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def data():
    fname = _download('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
    with _gzip.open(fname, 'rb') as f:
        if _sys.version_info[0] == 3:
            return _pickle.load(f, encoding='latin1')
        else:
            return _pickle.load(f)


def labels():
    return list(map(str, range(10)))
