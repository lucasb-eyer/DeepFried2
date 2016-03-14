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


def make_tensor_or_tensors(data_or_datas, name):
    if isinstance(data_or_datas, (list, tuple)):
        return [make_tensor(data.dtype, data.ndim, name + str(i+1)) for i, data in enumerate(data_or_datas)]
    else:
        return make_tensor(data_or_datas.dtype, data_or_datas.ndim, name)


def count_params(module, trainable_only=True):
    return sum(p.get_value().size for p in module.parameters(trainable_only=trainable_only))


def aslist(what):
    if isinstance(what, list):
        return what
    elif isinstance(what, tuple):
        return list(what)
    else:
        return [what]


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
