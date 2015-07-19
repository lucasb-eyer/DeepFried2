import numpy as _np

def xavier(shape, fan):
    assert fan is not None, "The parameter's `fan` needs to be specified when using Xavier initialization."

    w_bound = _np.sqrt(4. / sum(fan))
    return _np.random.uniform(low=-w_bound, high=w_bound, size=shape)
