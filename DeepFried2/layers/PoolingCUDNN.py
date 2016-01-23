import DeepFried2 as df


class PoolingCUDNN(df.Module):
    def __init__(self, window_size, stride=None, padding=None, mode='max'):
        """
        Wraps CUDNN's pooling operation.

        - `window_size`: A 2D or 3D tuple indicating the size of the pooling region.
        - `stride`: A tuple or number indicating the stride between poolings.
                    Defaults to `None`, which means `window_size`, i.e. non-overlapping.
        - `padding`: A tuple or number indicating the amount of 0-padding on either sides.
                     Defaults to `None` meaning no padding whatsoever.
        - `mode`: `'max'`, `'average_inc_pad'` or `'average_exc_pad'`.

        TODO: For now, you can do 1D pooling by using reshaping and setting
              width or height to 1, but we really should do that, or better yet
              PR to theano to allow 1D pooling (it seems cuDNN can).
        """
        df.Module.__init__(self)
        self.window_size = window_size
        self.mode = mode

        # Catch a probably common bug while we transition the API.
        assert isinstance(window_size, (list, tuple)), "New pooling API: window_size needs to be a tuple!"

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
            mode=self.mode,
            pad=self.padding
        )
