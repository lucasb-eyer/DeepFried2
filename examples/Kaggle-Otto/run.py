import numpy as np
import pandas as pd
import beacon8 as bb8
import beacon8.optimizers as optim
from os.path import dirname, join as pjoin
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from train import *
from test import *


def load_train_data():
    train_data = pd.read_csv(pjoin(dirname(__file__), 'data', 'train.csv'))
    labels = train_data.target.values
    labels_encoder = LabelEncoder()
    labels = labels_encoder.fit_transform(labels)
    train_data = train_data.drop('id', axis=1)
    train_data = train_data.drop('target', axis=1)
    return train_data.as_matrix(), labels

def nnet():
    model = bb8.Sequential()
    model.add(bb8.AddConstant(1.0))
    model.add(bb8.Log())
    model.add(bb8.BatchNormalization(93))
    model.add(bb8.Dropout(0.1))
    model.add(bb8.Linear(93, 512))
    model.add(bb8.BatchNormalization(512))
    model.add(bb8.ReLU())
    model.add(bb8.Dropout(0.5))

    model.add(bb8.Linear(512, 512))
    model.add(bb8.BatchNormalization(512))
    model.add(bb8.ReLU())
    model.add(bb8.Dropout(0.5))

    model.add(bb8.Linear(512, 512))
    model.add(bb8.BatchNormalization(512))
    model.add(bb8.ReLU())
    model.add(bb8.Dropout(0.5))

    model.add(bb8.Linear(512, 9))
    model.add(bb8.SoftMax())
    return model

if __name__ == "__main__":
    if __package__ is None:  # PEP366
        __package__ = "beacon8.examples.KaggleOtto"

    train_data_x, train_data_y = load_train_data()

    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(train_data_x, train_data_y, train_size=0.85)
    model = nnet()

    criterion = bb8.ClassNLLCriterion()

    optimiser = optim.Momentum(lr=0.01, momentum=0.9)

    for epoch in range(1000):
        model.training()
        if epoch > 100 and epoch % 100 == 0:
            optimiser.hyperparams['lr'] /= 10
        train(train_data_x, train_data_y, model, optimiser, criterion, epoch, 100)
        train(train_data_x, train_data_y, model, optimiser, criterion, epoch, 100, 'stat')

        model.evaluate()
        validate(test_data_x, test_data_y, model, epoch, 100)

