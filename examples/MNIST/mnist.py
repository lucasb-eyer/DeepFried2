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


def load_mnist(data_file = './mnist.pkl.gz'):

    if not os.path.exists(data_file):
        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print('Downloading data from %s' % origin)
        urlretrieve(origin, data_file)

    print('... loading data')

    f = gzip.open(data_file, 'rb')
    if sys.version_info[0] == 3:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    else:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)
