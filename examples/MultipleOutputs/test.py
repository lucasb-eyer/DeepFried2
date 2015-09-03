import DeepFried2 as df
import numpy as np

from examples.utils import make_progressbar


def validate(X, y_c, y_f, model, epoch, batch_size):
    progress = make_progressbar('Testing epoch #{}'.format(epoch), len(X))
    progress.start()

    nerrors_c, nerrors_f = 0, 0
    for ibatch in range((len(X) + batch_size - 1) // batch_size):
        # Note: numpy correctly handles the size of the last minibatch.
        Xbatch = X[ibatch*batch_size : (ibatch+1)*batch_size].astype(df.floatX)
        ybatch_c = y_c[ibatch*batch_size : (ibatch+1)*batch_size].astype(df.floatX)
        ybatch_f = y_f[ibatch*batch_size : (ibatch+1)*batch_size].astype(df.floatX)

        (preds_c, preds_f) = map(lambda py: np.argmax(py, axis=1), model.forward(Xbatch))

        nerrors_c += sum(ybatch_c != preds_c)
        nerrors_f += sum(ybatch_f != preds_f)

        progress.update(ibatch*batch_size + len(Xbatch))

    progress.finish()
    accuracy_c = 1 - float(nerrors_c)/len(X)
    accuracy_f = 1 - float(nerrors_f)/len(X)
    print("Epoch #{}, Coarse accuracy: {:.2%} ({} errors)".format(epoch, accuracy_c, nerrors_c))
    print("Epoch #{}, Fine   accuracy: {:.2%} ({} errors)".format(epoch, accuracy_f, nerrors_f))
