import DeepFried2 as df


def cnn(size, *head):
    return df.Sequential(
        df.Reshape(-1, 3, size, size),

        df.SpatialConvolutionCUDNN(3, 32, 3, 3, border='same', with_bias=False),
        df.BatchNormalization(32), df.ReLU(),
        df.SpatialConvolutionCUDNN(32, 32, 3, 3, border='same', with_bias=False),
        df.BatchNormalization(32), df.ReLU(),
        df.SpatialMaxPoolingCUDNN(2, 2),

        df.SpatialConvolutionCUDNN(32, 64, 3, 3, border='same', with_bias=False),
        df.BatchNormalization(64), df.ReLU(),
        df.SpatialConvolutionCUDNN(64, 64, 3, 3, border='same', with_bias=False),
        df.BatchNormalization(64), df.ReLU(),
        df.SpatialMaxPoolingCUDNN(2, 2),

        df.Reshape(-1, 64*(size//4)**2),

        df.Linear(64*(size//4)**2, 1024, with_bias=False),
        df.BatchNormalization(1024), df.ReLU(),
        df.Dropout(0.5),

        *head
    )
