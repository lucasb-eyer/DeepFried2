import numpy as np
import theano as th


def train(X, y, model, optimiser, criterion, batch_size, mode='train'):

    shuffle = np.random.permutation(len(X))

    for j in range(len(X) // batch_size):
        indices = shuffle[j*batch_size : (j+1)*batch_size]
        mini_batch_input = X[indices].astype(th.config.floatX)
        mini_batch_targets = y[indices].astype(th.config.floatX)

        if mode == 'train':
            model.zero_grad_parameters()
            model.accumulate_gradients(mini_batch_input, mini_batch_targets, criterion)
            optimiser.update_parameters(model)
        elif mode == 'stats':
            model.accumulate_statistics(mini_batch_input)
        else:
            assert False, "Mode should be either 'train' or 'stats'"
