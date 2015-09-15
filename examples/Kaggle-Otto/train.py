import DeepFried2 as df
import numpy as np

from examples.utils import make_progressbar


def train(dataset_x, dataset_y, model, optimiser, criterion, epoch, batch_size, mode='train'):
    progress = make_progressbar('Training ({}) epoch #{}'.format(mode, epoch), len(dataset_x))
    progress.start()

    shuffle = np.random.permutation(len(dataset_x))

    for j in range(dataset_x.shape[0] // batch_size):
        indices = shuffle[j*batch_size : (j+1)*batch_size]
        mini_batch_input = dataset_x[indices].astype(df.floatX)
        mini_batch_targets = dataset_y[indices].astype(df.floatX)

        if mode == 'train':
            model.zero_grad_parameters()
            model.accumulate_gradients(mini_batch_input, mini_batch_targets, criterion)
            optimiser.update_parameters(model)
        elif mode == 'stats':
            model.accumulate_statistics(mini_batch_input)
        else:
            assert False, "Mode should be either 'train' or 'stats'"

        progress.update(j*batch_size + len(mini_batch_input))

    progress.finish()
