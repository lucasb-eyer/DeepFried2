import os
import gzip
import pickle
import sys

# Python 2/3 compatibility.
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


'''Adapted from theano tutorial'''


def load_mnist(data_file = os.path.join(os.path.dirname(__file__), 'mnist.pkl.gz')):

    if not os.path.exists(data_file):
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from {}'.format(origin))
        urlretrieve(origin, data_file)

    print('... loading data')

    with gzip.open(data_file, 'rb') as f:
        if sys.version_info[0] == 3:
            return pickle.load(f, encoding='latin1')
        else:
            return pickle.load(f)
