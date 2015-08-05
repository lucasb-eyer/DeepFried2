import numpy as np
import pandas as pd
import DeepFried2 as df
import DeepFried2.optimizers as optim
from os.path import dirname, join as pjoin
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from train import train
from test import validate


def load_train_data():
    train_data = pd.read_csv(pjoin(dirname(__file__), 'data', 'train.csv'))
    labels = train_data.target.values
    labels = LabelEncoder().fit_transform(labels)
    train_data = train_data.drop('id', axis=1)
    train_data = train_data.drop('target', axis=1)
    return train_data.as_matrix(), labels

def nnet():
    model = df.Sequential()
    model.add(df.AddConstant(1.0))
    model.add(df.Log())
    model.add(df.BatchNormalization(93))
    model.add(df.Dropout(0.1))
    model.add(df.Linear(93, 512))
    model.add(df.BatchNormalization(512))
    model.add(df.ReLU())
    model.add(df.Dropout(0.5))

    model.add(df.Linear(512, 512))
    model.add(df.BatchNormalization(512))
    model.add(df.ReLU())
    model.add(df.Dropout(0.5))

    model.add(df.Linear(512, 512))
    model.add(df.BatchNormalization(512))
    model.add(df.ReLU())
    model.add(df.Dropout(0.5))

    model.add(df.Linear(512, 9))
    model.add(df.SoftMax())
    return model

if __name__ == "__main__":
    if __package__ is None:  # PEP366
        __package__ = "DeepFried2.examples.KaggleOtto"

    train_data_x, train_data_y = load_train_data()

    train_data_x, valid_data_x, train_data_y, valid_data_y = train_test_split(train_data_x, train_data_y, train_size=0.85)
    model = nnet()

    criterion = df.ClassNLLCriterion()

    optimiser = optim.Momentum(lr=0.01, momentum=0.9)

    for epoch in range(1, 1001):
        model.training()
        if epoch % 100 == 0:
            optimiser.hyperparams['lr'] /= 10
        train(train_data_x, train_data_y, model, optimiser, criterion, epoch, 100, 'train')
        train(train_data_x, train_data_y, model, optimiser, criterion, epoch, 100, 'stats')

        model.evaluate()
        validate(valid_data_x, valid_data_y, model, epoch, 100)

