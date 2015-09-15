import DeepFried2 as df
import numpy as np


def test(X, y, model, batch_size):

    nll = 0
    nerrors = 0
    for j in range((len(X) + batch_size - 1) // batch_size):
        # Note: numpy correctly handles the size of the last minibatch.
        miniX = X[j*batch_size : (j+1)*batch_size].astype(df.floatX)
        miniy = y[j*batch_size : (j+1)*batch_size]

        pred_probas = model.forward(miniX)
        preds = np.argmax(pred_probas, axis=1)

        nll -= sum(np.log(np.clip(pred_probas[np.arange(len(miniX)), miniy], 1e-15, 1-1e-15)))
        nerrors += sum(preds != miniy)

    #accuracy = 1 - float(nerrors)/len(X)
    return nll, nerrors
