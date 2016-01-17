import DeepFried2 as df


class SpatialMaxPoolingCUDNN(df.Module):
    def __init__(self, window_size, stride=None, padding=None):
        df.Module.__init__(self)
        self.window_size = window_size

        if stride is None:
            self.stride = window_size
        else:
            self.stride = stride

        if padding is None:
            self.padding = (0,)*len(window_size)
        else:
            self.padding = padding

    def symb_forward(self, symb_input):
        return df.th.sandbox.cuda.dnn.dnn_pool(
            img=symb_input,
            ws=self.window_size,
            stride=self.stride,
            mode='max',
            pad=self.padding
        )
