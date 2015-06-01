import numpy as np
import theano as _th
from sklearn.metrics import log_loss
from kaggle_utils import *

from examples.utils import make_progressbar

def validate(dataset_x, dataset_y, model, epoch, batch_size):
    progress = make_progressbar('Testing epoch #{}'.format(epoch), len(dataset_x))
    progress.start()

    mini_batch_input = np.empty(shape=(batch_size, 93), dtype=_th.config.floatX)
    mini_batch_targets = np.empty(shape=(batch_size, ), dtype=_th.config.floatX)
    accuracy = 0.

    for j in range((dataset_x.shape[0] + batch_size - 1) // batch_size):
        progress.update(j * batch_size)
        for k in range(batch_size):
            if j * batch_size + k < dataset_x.shape[0]:
                mini_batch_input[k] = dataset_x[j * batch_size + k]
                mini_batch_targets[k] = dataset_y[j * batch_size + k]

        mini_batch_prediction = model.forward(mini_batch_input)

        if (j + 1) * batch_size > dataset_x.shape[0]:
            mini_batch_prediction.resize((dataset_x.shape[0] - j * batch_size, 9))
            mini_batch_targets.resize((dataset_x.shape[0] - j * batch_size, ))

        accuracy = accuracy + multiclass_log_loss(mini_batch_targets, mini_batch_prediction, normalize=False)

    progress.finish()
    print("Epoch #" + str(epoch) + ", Logloss: " + str(float(accuracy) / dataset_x.shape[0]))
