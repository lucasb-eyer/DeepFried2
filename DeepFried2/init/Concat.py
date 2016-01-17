import numpy as _np

def concat(*inits, axis=-1):
    """
    Initializes a tensor as a concatenation of the given `inits` along the
    given `axis`, which defaults to the last one.

    Useful for creating e.g. "merged dots" as in GRU/LSTM.
    """

    def init(shape, fan):
        N = len(inits)
        assert shape[axis] % N == 0, "Number of inits ({}) doesn't divide shape ({}) on given axis ({})".format(N, shape, axis)
        subshape = list(shape)
        subfan = list(fan)
        subshape[axis] //= N
        subfan[axis] //= N
        return _np.concatenate([i(subshape, subfan) for i in inits], axis=axis)

    return init
