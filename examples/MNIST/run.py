import DeepFried2 as df

from train import train
from test import validate
from model import net, lenet, lenet_same, lenet_cudnn


def main(params):
    train_set, valid_set, test_set = df.datasets.mnist.data()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set

    model = lenet()
    criterion = df.ClassNLLCriterion()
    optimiser = df.SGD(lr=params['lr'])

    print("Compiling model...")
    for epoch in range(100):
        model.training()
        train(train_set_x, train_set_y, model, optimiser, criterion, epoch, params['batch_size'], 'train')

        train(train_set_x, train_set_y, model, optimiser, criterion, epoch, params['batch_size'], 'stats')
        model.evaluate()
        validate(test_set_x, test_set_y, model, epoch, params['batch_size'])


if __name__ == "__main__":
    if __package__ is None:  # PEP366
        __package__ = "DeepFried2.examples.MNIST"

    params = {}
    params['lr'] = 0.1
    params['batch_size'] = 64
    main(params)
