import DeepFried2 as df
import numpy as _np
from warnings import warn as _warn
from numbers import Number as _Number


def create_param_state_as(other, initial_value=0, prefix='state_for_'):
    return df.th.shared(other.get_value()*0 + initial_value,
        broadcastable=other.broadcastable,
        name=prefix + str(other.name)
    )


def _check_dtype_mistake(dtype):
    """
    It's a very common mistake (at least for me) to pass-in a float64 when I
    really want to pass in a `floatX`, and it would go unnoticed and slow-down
    the computations a lot if I wouldn't check it here.
    """
    if _np.issubdtype(dtype, _np.floating) and dtype != df.floatX:
        _warn("Input array of floating-point dtype {} != df.floatX detected. Is this really what you want?".format(dtype))


def make_tensor(dtype, ndim, name):
    _check_dtype_mistake(dtype)
    return df.th.tensor.TensorType(dtype, (False,) * ndim)(name)


def tensors_for_ndarrays(datas, basename):
    #if isinstance(datas, _np.ndarray):
    try:
        dtype, ndim = datas.dtype, datas.ndim
    except AttributeError:
        pass  # Not an array-like
    else:
        return make_tensor(dtype, ndim, basename)

    if isinstance(datas, (list, tuple)):
        return [tensors_for_ndarrays(data, "{}_{}".format(basename, i)) for i, data in enumerate(datas)]
    # Could potentially make it "any iterable" by removing above check.
    # But would need to guarantee we never iterate over it twice, which is harder!
    raise TypeError("I only understand lists or tuples of numpy arrays! (possibly nested)")


def count_params(module, learnable_only=True):
    return sum(p.get_value().size for p in module.parameters(learnable_only=learnable_only))


def reinit(module):
    for p in module.parameters():
        p.reinit()
    return module


def freeze(module):
    for p in module.parameters(learnable_only=True):
        p.freeze()
    return module


def thaw(module):
    for p in module.parameters():
        p.thaw()
    return module


def flatten(what, types=(list, tuple), none_to_empty=False):
    if what is None and none_to_empty:
        return []

    if not isinstance(what, types):
        return [what]

    # NOTE: I actually timed that this is faster than the comprehension,
    #       even though it probably doesn't matter :)
    # 350us vs 250us
    ret = []
    for sub in what:
        ret += flatten(sub, types=types, none_to_empty=none_to_empty)
    return ret


def expand(tup, ndim, name=None, expand_nonnum=False):
    if isinstance(tup, (tuple, list)) and len(tup) == ndim:
        return tup

    if isinstance(tup, _Number) or expand_nonnum:
        return (tup,) * ndim

    if not expand_nonnum:
        return tup

    raise ValueError("Bad number of dimensions{}: is {} but should be {}.".format((" for " + name) if name else "", len(tup), ndim))

def typename(obj):
    return type(obj).__name__


def pad(symb_input, padding):
    assert symb_input.ndim == len(padding), "symb_input ({}d) and padding ({}d) must have the same dimensionality".format(symb_input.ndim, len(padding))

    padded_shape = tuple((s+2*p) for s,p in zip(symb_input.shape, padding))
    padded_input = df.T.zeros(padded_shape)

    slicing = [slice(None) if p == 0 else slice(p,s+p) for s,p in zip(symb_input.shape, padding)]

    return df.T.set_subtensor(padded_input[slicing], symb_input)
