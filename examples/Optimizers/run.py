import DeepFried2 as df

from examples.utils import make_progressbar

from mnist import load_mnist
from train import train
from test import test
from model import net, lenet2


if __name__ == "__main__":
    print("THIS IS JUST AN EXAMPLE.")
    print("Please don't take these numbers as a benchmark.")
    print("While the optimizer's parameters have been grid-searched,")
    print("a fair comparison would run all experiments multiple times AND RUN MORE THAN FIVE EPOCHS.")

    batch_size = 64

    (Xtrain, ytrain), (Xval, yval), (Xtest, ytest) = load_mnist()

    criterion = df.ClassNLLCriterion()

    def run(optim):
        progress = make_progressbar('Training with ' + str(optim), 5)
        progress.start()

        model = net()
        model.training()
        for epoch in range(5):
            train(Xtrain, ytrain, model, optim, criterion, batch_size, 'train')
            train(Xtrain, ytrain, model, optim, criterion, batch_size, 'stats')
            progress.update(epoch+1)

        progress.finish()

        model.evaluate()
        nll, _ = test(Xtrain, ytrain, model, batch_size)
        _, nerr = test(Xval, yval, model, batch_size)

        print("Trainset NLL: {:.2f}".format(nll))
        print("Testset errors: {}".format(nerr))

    run(df.SGD(lr=1e-1))
    run(df.Momentum(lr=1e-2, momentum=0.95))
    run(df.Nesterov(lr=1e-2, momentum=0.90))
    run(df.AdaGrad(lr=1e-2, eps=1e-4))
    run(df.RMSProp(lr=1e-3, rho=0.90, eps=1e-5))
    run(df.AdaDelta(rho=0.99, lr=5e-1, eps=1e-4))
