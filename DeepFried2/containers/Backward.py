import DeepFried2 as df


def backward(start, end):
    """
    Returns a function that, when executed, sends its argument backwards
    through the (sub-)graph that goes from `start` to `end`.
    Useful e.g. for bwd-conv and un-pooling.

    Please give credit when simply copy-pasting into your code.
    """
    return lambda x: df.th.grad(None, wrt=start, known_grads={end: x})


class Backward(df.SingleModuleContainer):
    """
    Uses the backward-pass of the contained `Module` as forward pass.

    For example, to get an un-pooling layer:

        p = df.PoolingCUDNN((2,2))
        u = df.Backward(p)
        net = df.Sequential(..., p, ..., u, ...)

    Or a "deconvolution", better called backward-convolution:

        c = df.SpatialConvolutionCUDNN(n_in, n_out, (3,3))
        d = df.Backward(c)
        net = df.Sequential(..., c, ..., d, ...)

    Note that in this case, the convolution and backward-convolution share
    the same weights! This might not be what you want.

    If you don't want to share weights, you need to create a second `Module`
    which shall be used for the backward-pass, but "relate" it to the original
    module that it should "undo" using the `wrt` keyword argument:

        c = df.SpatialConvolutionCUDNN(n_in, n_out, (3,3))
        d = df.Backward(df.SpatialConvolutionCUDNN(n_in, n_out, (3,3)), wrt=c)
        net = df.Sequential(..., c, ..., d, ...)

    In this case, each have their own set of independent weights.

    NOTE: The contained module (or that in `wrt`, if given) must appear in the
          network's graph before the `Backward` version of it.
    """
    def __init__(self, module, wrt=None):
        df.SingleModuleContainer.__init__(self, module)
        self.wrt = wrt

    def symb_forward(self, symb_input):
        # If no `wrt` is passed, we use the referenced module's graph and go
        # through it backwards, using all its parameters etc.
        if self.wrt is None:
            try:
                start = self.modules[0]._last_symb_inp[self._mode]
                end   = self.modules[0]._last_symb_out[self._mode]
            except KeyError:
                raise ValueError("The module contained by `Backward` needs to occur first in the graph!")

        # But if `wrt` is passed, we call `symb_forward` of the referenced
        # module with the input that `wrt` had gotten, in order to create a
        # new graph through which we then go backwards.
        else:
            try:
                start = self.wrt._last_symb_inp[self._mode]
            except KeyError:
                raise ValueError("The module referenced by `Backward`'s `wrt` argument needs to occur first in the graph!")
            end = self.modules[0](start)

        end = df.utils.flatten(end)
        inp = df.utils.flatten(symb_input)

        # Match all "backward outputs" with inputs here.
        assert len(end) == len(inp), "Need same number of inputs to `Backward` as contained module has outputs ({})".format(len(end))
        known_grads = dict(zip(end, inp))

        # Go backwards for each "backward input" which then becomes output here.
        if isinstance(start, (list, tuple)):
            return [df.th.grad(None, wrt=s, known_grads=known_grads) for s in start]
        else:
            return df.th.grad(None, wrt=start, known_grads=known_grads)
