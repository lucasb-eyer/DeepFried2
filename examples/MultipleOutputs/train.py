import DeepFried2 as df
import numpy as np

from examples.utils import make_progressbar


def train(X, y_c, y_f, model, optim, crit, epoch, batch_size, mode='train'):
    progress = make_progressbar('Training ({}) epoch #{}'.format(mode, epoch), len(X))
    progress.start()

    shuffle = np.random.permutation(len(X))

    for ibatch in range(len(X) // batch_size):
        indices = shuffle[ibatch*batch_size : (ibatch+1)*batch_size]
        Xbatch = X[indices].astype(df.floatX)
        ybatch_c = y_c[indices].astype(df.floatX)
        ybatch_f = y_f[indices].astype(df.floatX)

        if mode == 'train':
            model.zero_grad_parameters()
            model.accumulate_gradients(Xbatch, (ybatch_c, ybatch_f), crit)
            optim.update_parameters(model)
        elif mode == 'stats':
            model.accumulate_statistics(Xbatch)
        else:
            assert False, "Mode should be either 'train' or 'stats'"

        progress.update(ibatch*batch_size + len(Xbatch))

    progress.finish()
