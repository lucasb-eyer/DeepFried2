from . import Module

import numpy as _np
import theano as _th

class Linear(Module):

    def __init__(self, nin, nout, init='Xavier', with_bias=True):
        super().__init__()

        self.nin = nin
        self.nout = nout
        self.init = init
        self.with_bias = with_bias

        self.reset()

    def reset(self):
        if self.init == 'Xavier':
            w_bound = _np.sqrt(4 / (self.nin + self.nout))
            W = _np.random.uniform(low=-w_bound, high=w_bound,
                                   size=(self.nin, self.nout))
        else:
            raise NotImplementedError

        self.weight = _th.shared(W.astype(_th.config.floatX))

        if self.with_bias:
            self.bias = _th.shared(_np.zeros(shape=self.nout, dtype=_th.config.floatX))

    def symbolic_forward(self, symbolic_input):
        out = _th.tensor.dot(symbolic_input, self.weight)

        if self.with_bias:
            out += self.bias

        return out
