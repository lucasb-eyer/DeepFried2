import DeepFried2 as df


class UpSample(df.Module):
    def __init__(self, upsample=(2,2), axes=[-2, -1], output_shape=None, mode='repeat'):
        """
        Upsamples an input (nearest-neighbour, repeat) `upsample` times along `axes`.

        - `axes` is a tuple specifying the dimensions along which to upsample.
        - `upsample` is a tuple of the same length as `axes` specifying the integer upsampling factor along each corresponding axis.
            If all `axes` should be upsampled by the same factor, `upsample` can also be just that factor.
        - `output_shape` can be used to crop the upsampled result to a desired shape, it follows `axes` just like `upsample` does.
            (TODO: move this part into a separate `Restrict` module)
        - `mode` specifies how upsampling happens. Currently supported are:
            - `repeat`: upsample by repeating values, aka "nearest" (the default).
            - `perforated`: Put values in top-left corner, i.e. [1,2] becomes [1,0,2,0].
                            This comes from http://www.brml.org/uploads/tx_sibibtex/281.pdf
        """
        df.Module.__init__(self)
        self.axes = axes
        self.upsample = df.utils.expand(upsample, len(axes), "upsample factor")
        self.output_shape = df.utils.expand(output_shape, len(axes), "output shape")
        self.upsample_mode = mode

    def symb_forward(self, symb_input):
        """symb_input shape: 2D: (n_input, channels, height, width)
                             3D: (n_input, channels, depth, height, width)
        """
        if self.upsample_mode == 'repeat':
            res = symb_input
            for f, ax in zip(self.upsample, self.axes):
                res = df.T.repeat(res, f, axis=ax)
        elif self.upsample_mode == 'perforated':
            shape = list(symb_input.shape)
            slices = [slice(None)]*symb_input.ndim
            for f, ax in zip(self.upsample, self.axes):
                shape[ax] *= f
                slices[ax] = slice(None, None, f)
            res = df.T.zeros(shape, symb_input.dtype)
            res = df.T.set_subtensor(res[tuple(slices)], symb_input)
        else:
            raise ValueError("Unsupported upsampling mode '{}'".format(self.upsample_mode))

        # TODO: move this out to its own `Restrict` module.
        if self.output_shape is not None:
            restrict = [slice(None)]*symb_input.ndim
            for sh, ax in zip(self.output_shape, self.axes):
                restrict[ax] = slice(sh)
            res = res[tuple(restrict)]

        return res
