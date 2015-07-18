import DeepFried2 as df


def net():
    model = df.Sequential()
    model.add(df.Linear(28*28, 100))
    model.add(df.ReLU())

    model.add(df.Linear(100, 100))
    model.add(df.ReLU())

    model.add(df.Linear(100, 100))
    model.add(df.ReLU())

    model.add(df.Linear(100, 10))
    model.add(df.SoftMax())
    return model


def lenet():
    model = df.Sequential()
    model.add(df.Reshape(-1, 1, 28, 28))
    model.add(df.SpatialConvolutionCUDNN(1, 32, 5, 5, 1, 1, 2, 2, with_bias=False))
    model.add(df.BatchNormalization(32))
    model.add(df.ReLU())
    model.add(df.SpatialMaxPoolingCUDNN(2, 2))

    model.add(df.SpatialConvolutionCUDNN(32, 64, 5, 5, 1, 1, 2, 2, with_bias=False))
    model.add(df.BatchNormalization(64))
    model.add(df.ReLU())
    model.add(df.SpatialMaxPoolingCUDNN(2, 2))
    model.add(df.Reshape(-1, 7*7*64))

    model.add(df.Linear(7*7*64, 100, with_bias=False))
    model.add(df.BatchNormalization(100))
    model.add(df.ReLU())
    model.add(df.Dropout(0.5))

    model.add(df.Linear(100, 10))
    model.add(df.SoftMax())
    return model


def lenet2():
    model = df.Sequential()
    model.add(df.Reshape(-1, 1, 28, 28))
    model.add(df.SpatialConvolution(1, 32, 5, 5, 1, 1, with_bias=False))
    model.add(df.BatchNormalization(32))
    model.add(df.ReLU())
    model.add(df.SpatialMaxPooling(2, 2))

    model.add(df.SpatialConvolution(32, 64, 5, 5, 1, 1, with_bias=False))
    model.add(df.BatchNormalization(64))
    model.add(df.ReLU())
    model.add(df.SpatialMaxPooling(2, 2))
    model.add(df.Reshape(-1, 4*4*64))

    model.add(df.Linear(4*4*64, 100, with_bias=False))
    model.add(df.BatchNormalization(100))
    model.add(df.ReLU())
    model.add(df.Dropout(0.5))

    model.add(df.Linear(100, 10))
    model.add(df.SoftMax())
    return model
