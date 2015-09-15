import DeepFried2 as df
import numpy as np

from examples.utils import make_progressbar
from itertools import combinations

def validate(Xdata, ydata, model, epoch, batch_size):
    progress = make_progressbar('Testing epoch #{}'.format(epoch), len(Xdata))
    progress.start()

    nerrors = 0
    ntrials = 0

    # In theory, we should check all pairs.
    # But for MNIST, that'd be roughly 10k**2/2 ~ 50M
    # which is too much for this example.

    # So instead, we do all pairs in each batch, which for a batch of 100
    # is ~ 100**2/2 * 10k/100 ~ 500k

    for ibatch in range((len(Xdata) + batch_size - 1) // batch_size):
        # Note: numpy correctly handles the size of the last minibatch.
        batch_indices = np.arange(ibatch*batch_size, min((ibatch+1)*batch_size, len(Xdata)))

        batchleft, batchright = map(list, zip(*combinations(batch_indices, r=2)))

        # But then, for each batch we've got 100**2/2 ~ 5k pairs,
        # thus we need to go through them in batch-mode again (cuz GPU memory)

        for i in range((len(batchleft) + batch_size - 1) // batch_size):
            Xleft  = Xdata[batchleft [i*batch_size : (i+1)*batch_size]].astype(df.floatX)
            yleft  = ydata[batchleft [i*batch_size : (i+1)*batch_size]]
            Xright = Xdata[batchright[i*batch_size : (i+1)*batch_size]].astype(df.floatX)
            yright = ydata[batchright[i*batch_size : (i+1)*batch_size]]

            preds = 0.5 < np.squeeze(model.forward((Xleft, Xright)))
            nerrors += sum(preds != (yleft == yright))
            ntrials += len(preds)

        progress.update(ibatch*batch_size + len(batch_indices))

    progress.finish()
    accuracy = 1 - float(nerrors)/ntrials
    print("Epoch #{}, Classification accuracy: {:.2%} ({} errors out of {} tests)".format(epoch, accuracy, nerrors, ntrials))
