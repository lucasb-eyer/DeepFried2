import DeepFried2 as df
import numpy as np

from kaggle_utils import multiclass_log_loss
from examples.utils import make_progressbar

def validate(dataset_x, dataset_y, model, epoch, batch_size):
    progress = make_progressbar('Testing epoch #{}'.format(epoch), len(dataset_x))
    progress.start()

    logloss = 0.
    for j in range((dataset_x.shape[0] + batch_size - 1) // batch_size):
        # Note: numpy correctly handles the size of the last minibatch.
        mini_batch_input = dataset_x[j*batch_size : (j+1)*batch_size].astype(df.floatX)
        mini_batch_targets = dataset_y[j*batch_size : (j+1)*batch_size]

        mini_batch_prediction = model.forward(mini_batch_input)

        logloss += multiclass_log_loss(mini_batch_targets, mini_batch_prediction, normalize=False)

        progress.update(j * batch_size + len(mini_batch_input))

    progress.finish()
    print("Epoch #{}, Logloss: {:.5f}".format(epoch, logloss/dataset_x.shape[0]))
