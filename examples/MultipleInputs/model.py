import DeepFried2 as df


def twinnet():
    tunnel = df.Sequential(
        df.Reshape(-1, 1, 28, 28),
        df.SpatialConvolution(1, 32, (5,5), with_bias=False),
        df.BatchNormalization(32),
        df.ReLU(),
        df.SpatialMaxPooling((2,2)),

        df.SpatialConvolution(32, 64, (5,5), with_bias=False),
        df.BatchNormalization(64),
        df.ReLU(),
        df.SpatialMaxPooling((2,2)),
        df.Reshape(-1, 4*4*64),
    )

    return df.Sequential(
        df.Parallel(
            tunnel,
            tunnel
        ),

        df.Concat(axis=1),

        df.Linear(2*4*4*64, 100, with_bias=False),
        df.BatchNormalization(100),
        df.ReLU(),
        df.Dropout(0.5),

        df.Linear(100, 1),
        df.Sigmoid()
    )
