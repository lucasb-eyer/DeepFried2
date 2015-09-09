import DeepFried2 as df

from train import *
from test import *
from model import *


def main(params):
    # c ^= coarse, f ^= fine
    (Xtr, ytr_c, ytr_f), (Xva, yva_c, yva_f), (Xte, yte_c, yte_f), (le_c, le_f) = df.datasets.cifar100.data()

    model = cnn(32,
        df.Parallel(
            df.Sequential(df.Linear(1024, 20, initW=df.init.const(0)), df.SoftMax()),
            df.Sequential(df.Linear(1024,100, initW=df.init.const(0)), df.SoftMax()),
        )
    )

    crit = df.ParallelCriterion(
        (1, df.ClassNLLCriterion()),
        (1, df.ClassNLLCriterion()),
        penalties=[
            (1e-4, df.L1WeightDecay(model)),
            (1e-5, df.L2WeightDecay(model)),
        ]
    )

    optim = df.AdaDelta(params['adadelta_rho'])

    for epoch in range(100):
        model.training()
        train(Xtr, ytr_c, ytr_f, model, optim, crit, epoch, params['batch_size'], 'train')

        train(Xtr, ytr_c, ytr_f, model, optim, crit, epoch, params['batch_size'], 'stats')
        model.evaluate()
        validate(Xva, yva_c, yva_f, model, epoch, params['batch_size'])


if __name__ == "__main__":
    if __package__ is None:  # PEP366
        __package__ = "DeepFried2.examples.MultipleOutputs"

    params = {}
    params['adadelta_rho'] = 0.95
    params['batch_size'] = 128
    main(params)
