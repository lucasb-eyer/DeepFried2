import DeepFried2 as df
import DeepFried2.optimizers as optim

from train import *
from test import *
from model import *


def main(params):
    (Xtrain, ytrain), (Xval, yval), (Xtest, ytest) = df.datasets.mnist.data()

    model = twinnet()
    #criterion = df.ClassNLLCriterion()
    criterion = df.BCECriterion()
    optimiser = df.optimizers.AdaDelta(rho=0.95)

    for epoch in range(100):
        model.training()
        train(Xtrain, ytrain, model, optimiser, criterion, epoch, params['batch_size'], 'train')

        if epoch % 3 == 0:
            train(Xtrain, ytrain, model, optimiser, criterion, epoch, params['batch_size'], 'stats')
            model.evaluate()
            validate(Xval, yval, model, epoch, params['batch_size'])


if __name__ == "__main__":
    if __package__ is None:  # PEP366
        __package__ = "DeepFried2.examples.MultipleInputs"

    params = {}
    params['lr'] = 0.1
    params['batch_size'] = 64
    main(params)
