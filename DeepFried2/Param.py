import DeepFried2 as df
import numpy as _np


class Param(object):

    def __init__(self, shape, init, fan=None, name=None, learn=True, decay=True, dtype=df.floatX, **kw):
        self.init = init
        self.shape = (shape,) if _np.isscalar(shape) else tuple(shape)
        self.fan = fan
        self.decay = decay

        # Support a couple useful shortcut for initializing:
        if hasattr(init, 'shape') and hasattr(init, 'dtype'):
            # TODO: It would be nicer to use Python's buffer-interface.
            self.init = df.init.array(init)
        elif _np.isscalar(init):
            self.init = df.init.const(init)

        val = self.init(self.shape, self.fan).astype(dtype)
        self.param = df.th.shared(val, name=name, **kw)

        if learn:
            grad_name = 'grad_' + name if name is not None else None
            self.grad = df.th.shared(_np.zeros_like(val), name=grad_name, **kw)
        else:
            self.grad = None

    def get_value(self):
        return self.param.get_value()

    def set_value(self, val):
        self.param.set_value(val)

    def reinit(self):
        self.param.set_value(self.init(self.shape, self.fan).astype(self.param.dtype))

    def zero_grad(self):
        self.grad.set_value(_np.zeros(self.shape, self.param.dtype))

    def may_decay(self):
        return self.grad is not None and self.decay

    def trainable(self):
        return self.grad is not None
