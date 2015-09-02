import DeepFried2 as df
import numpy as np

from examples.utils import make_progressbar


def train(Xtrain, ytrain, model, optimiser, criterion, epoch, batch_size, mode='train'):
    progress = make_progressbar('Training ({}) epoch #{}'.format(mode, epoch), len(Xtrain))
    progress.start()

    shuffle = np.random.permutation(len(Xtrain))

    for ibatch in range(len(Xtrain) // 2 // batch_size):
        indices = shuffle[ibatch*batch_size*2 : (ibatch+1)*batch_size*2]

        Xleft = Xtrain[indices[:batch_size]].astype(df.floatX)
        yleft = ytrain[indices[:batch_size]].astype(df.floatX)
        Xright = Xtrain[indices[batch_size:]].astype(df.floatX)
        yright = ytrain[indices[batch_size:]].astype(df.floatX)

        # Need to put the targets into a column because of the way BCE works.
        y = (yleft == yright)[:,None].astype(df.floatX)

        if mode == 'train':
            model.zero_grad_parameters()
            model.accumulate_gradients((Xleft, Xright), y, criterion)
            optimiser.update_parameters(model)
        elif mode == 'stats':
            model.accumulate_statistics((Xleft, Xright))
        else:
            assert False, "Mode should be either 'train' or 'stats'"

        progress.update(ibatch*batch_size*2 + len(y))

    progress.finish()
