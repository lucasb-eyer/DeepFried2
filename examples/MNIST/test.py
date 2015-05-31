import numpy as np
import theano as th

from examples.utils import make_progressbar

def validate(dataset_x, dataset_y, model, epoch, batch_size):
    progress = make_progressbar('Testing epoch #{}'.format(epoch), len(dataset_x))
    progress.start()

    nerrors = 0
    for j in range((dataset_x.shape[0] + batch_size - 1) // batch_size):
        # Note: numpy correctly handles the size of the last minibatch.
        mini_batch_input = dataset_x[j*batch_size : (j+1)*batch_size].astype(th.config.floatX)
        mini_batch_targets = dataset_y[j*batch_size : (j+1)*batch_size].astype(th.config.floatX)

        mini_batch_prediction = np.argmax(model.forward(mini_batch_input), axis=1)

        nerrors += sum(mini_batch_targets != mini_batch_prediction)

        progress.update(j * batch_size)

    progress.finish()
    accuracy = 1 - float(nerrors)/dataset_x.shape[0]
    print("Epoch #{}, Classification accuracy: {:.2%} ({} errors)".format(epoch, accuracy, nerrors))
