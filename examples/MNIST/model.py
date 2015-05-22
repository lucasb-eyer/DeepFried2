import beacon8 as bb8


def net():
    model = bb8.Sequential()
    model.add(bb8.Linear(28*28, 100))
    model.add(bb8.ReLU())

    model.add(bb8.Linear(100, 100))
    model.add(bb8.ReLU())

    model.add(bb8.Linear(100, 100))
    model.add(bb8.ReLU())

    model.add(bb8.Linear(100, 10))
    model.add(bb8.SoftMax())
    return model


def lenet():
    model = bb8.Sequential()
    model.add(bb8.Reshape(-1, 1, 28, 28))
    model.add(bb8.SpatialConvolutionCUDNN(1, 32, 5, 5, 1, 1, 2, 2, with_bias=False))
    model.add(bb8.BatchNormalization(32))
    model.add(bb8.ReLU())
    model.add(bb8.SpatialMaxPoolingCUDNN(2, 2))

    model.add(bb8.SpatialConvolutionCUDNN(32, 64, 5, 5, 1, 1, 2, 2, with_bias=False))
    model.add(bb8.BatchNormalization(64))
    model.add(bb8.ReLU())
    model.add(bb8.SpatialMaxPoolingCUDNN(2, 2))
    model.add(bb8.Reshape(-1, 7*7*64))

    model.add(bb8.Linear(7*7*64, 100, with_bias=False))
    model.add(bb8.BatchNormalization(100))
    model.add(bb8.ReLU())
    model.add(bb8.Dropout(0.5))

    model.add(bb8.Linear(100, 10))
    model.add(bb8.SoftMax())
    return model

