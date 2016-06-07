import DeepFried2 as df


class UpSample(df.Module):
    def __init__(self, upsample=(2,2), axes=[-2, -1], output_shape=None):
        """
        Upsamples an input (nearest-neighbour, repeat) `upsample` times along `axes`.

        - `axes` is a tuple specifying the dimensions along which to upsample.
        - `upsample` is a tuple of the same length as `axes` specifying the integer upsampling factor along each corresponding axis.
            If all `axes` should be upsampled by the same factor, `upsample` can also be just that factor.
        - `output_shape` can be used to crop the upsampled result to a desired shape, it follows `axes` just like `upsample` does.
            (TODO: move this part into a separate `Restrict` module)
        """
        df.Module.__init__(self)
        self.axes = axes
        self.upsample = df.utils.expand(upsample, len(axes), "upsample factor")
        self.output_shape = df.utils.expand(output_shape, len(axes), "output shape")

    def symb_forward(self, symb_input):
        """symb_input shape: 2D: (n_input, channels, height, width)
                             3D: (n_input, channels, depth, height, width)
        """
        res = symb_input
        for f, ax in zip(self.upsample, self.axes):
            res = df.T.repeat(res, f, axis=ax)

        # TODO: move this out to its own `Restrict` module.
        if self.output_shape is not None:
            restrict = [slice(None)]*symb_input.ndim
            for sh, ax in zip(self.output_shape, self.axes):
                restrict[ax] = slice(sh)
            res = res[tuple(restrict)]

        return res
