import DeepFried2 as df
import numpy as _np


def create_param(shape, init, fan=None, name=None, type=df.floatX):
    return df.th.shared(init(shape, fan).astype(type), name=name)


def create_param_and_grad(shape, init, fan=None, name=None, type=df.floatX):
    val = init(shape, fan).astype(type)
    param = df.th.shared(val, name=name)
    grad_name = 'grad_' + name if name is not None else None
    grad_param = df.th.shared(_np.zeros_like(val), name=grad_name)
    return param, grad_param


def create_param_state_as(other, initial_value=0, prefix='state_for_'):
    return df.th.shared(other.get_value()*0 + initial_value,
        broadcastable=other.broadcastable,
        name=prefix + str(other.name)
    )


def make_tensor(ndim, name):
    return df.th.tensor.TensorType(df.floatX, (False,) * ndim)(name)


def make_tensor_or_tensors(data_or_datas, name):
    if isinstance(data_or_datas, (list, tuple)):
        return [make_tensor(data.ndim, name + str(i+1)) for i, data in enumerate(data_or_datas)]
    else:
        return make_tensor(data_or_datas.ndim, name)


def count_params(module):
    params, _ = module.unique_parameters()
    return sum(p.get_value().size for p in params)


def save_params(module, where, compress=False):
    params, _ = module.unique_parameters()

    savefn = _np.savez_compressed if compress else _np.savez
    savefn(where, params=[p.get_value() for p in params])


def load_params(module, fromwhere):
    params, _ = module.unique_parameters()
    with _np.load(fromwhere) as f:
        for p, v in zip(params, f['params']):
            p.set_value(v)


def aslist(what):
    if isinstance(what, list):
        return what
    elif isinstance(what, tuple):
        return list(what)
    else:
        return [what]


def typename(obj):
    return type(obj).__name__
