import DeepFried2 as df
import numpy as _np


class Param(object):

    def __init__(self, shape, init, fan=None, name=None, learn=True, decay=True, dtype=df.floatX, **kw):
        self.init = init
        self.shape = (shape,) if _np.isscalar(shape) else tuple(shape)
        self.fan = fan
        self.decay = decay
        self._kw = kw

        # Support a couple useful shortcut for initializing:
        if hasattr(init, 'shape') and hasattr(init, 'dtype'):
            # TODO: It would be nicer to use Python's buffer-interface.
            self.init = df.init.array(init)
        elif _np.isscalar(init):
            self.init = df.init.const(init)

        val = self.init(self.shape, self.fan).astype(dtype)
        self.param = df.th.shared(val, name=name, **kw)

        if learn:
            self.thaw()
        else:
            self.freeze()

    def get_value(self):
        return self.param.get_value()

    def set_value(self, val):
        self.param.set_value(val)

    def reinit(self):
        self.param.set_value(self.init(self.shape, self.fan).astype(self.param.dtype))

    def zero_grad(self):
        if self.learnable():
            self.grad.set_value(_np.zeros(self.shape, self.param.dtype))

    def may_decay(self):
        return self.learnable() and self.decay

    def learnable(self):
        return self.grad is not None

    def thaw(self):
        grad_name = 'grad_' + self.param.name if self.param.name is not None else None
        self.grad = df.th.shared(_np.zeros_like(self.param.get_value()), name=grad_name, **self._kw)

    def freeze(self):
        self.grad = None
