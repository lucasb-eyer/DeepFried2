import beacon8.optimizers as optim
from mnist import *
from train import *
from test import *
from model import *


def main(params):
    train_set, valid_set, test_set = load_mnist()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set

    model = lenet()

    criterion = bb8.ClassNLLCriterion()

    optimiser = optim.SGD(lr=params['lr'])

    for epoch in range(100):
        model.training()
        train(train_set_x, train_set_y, model, optimiser, criterion, epoch, params['batch_size'])
        train(train_set_x, train_set_y, model, optimiser, criterion, epoch, params['batch_size'], 'stat')

        model.evaluate()
        validate(test_set_x, test_set_y, model, epoch, params['batch_size'])


if __name__ == "__main__":
    params = {}
    params['lr'] = 0.1
    params['batch_size'] = 64
    main(params)