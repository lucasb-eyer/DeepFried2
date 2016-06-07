import DeepFried2 as df
from .images import imread

import numpy as _np
from tarfile import open as _taropen
import os.path as _p


def data(fold=False):
    fname = df.zoo.download('http://dags.stanford.edu/data/iccv09Data.tar.gz')

    # extracting files one-by-one in memory is unfortunately WAY too slow
    # for this dataset. So we bite the bullet and extract the full tgz.

    where = _p.dirname(fname)
    imgdir = 'iccv09Data/images/'

    with _taropen(fname, 'r') as f:
        f.extractall(where)
        ids = [_p.basename(n)[:-4] for n in f.getnames() if n.startswith(imgdir)]

    X = [imread(_p.join(where, imgdir, i) + '.jpg') for i in ids]
    y = [_np.loadtxt(_p.join(where, 'iccv09Data/labels', i) + '.regions.txt', dtype=_np.int32) for i in ids]
    # I personally don't believe in the other label types.

    le = _np.array(['sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain', 'foreground', 'object'])
    try:
        from sklearn.preprocessing import LabelEncoder
        le, classes = LabelEncoder(), le
        le.classes_ = classes
    except ImportError:
        pass

    if fold is False:
        return X, y, le

    lo, hi = fold*ntest(), (fold+1)*ntest()
    Xtr = X[:lo] + X[hi:]
    ytr = y[:lo] + y[hi:]
    Xte = X[lo:hi]
    yte = y[lo:hi]
    return (Xtr, ytr), (Xte, yte), le


def nfold():
    return 5


def ntrain():
    return 572


def ntest():
    return 143
