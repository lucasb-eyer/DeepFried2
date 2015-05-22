from .Module import Module

import numpy as _np
import theano as _th


class Linear(Module):

    def __init__(self, nin, nout, init='Xavier', with_bias=True):
        Module.__init__(self)

        self.nin = nin
        self.nout = nout
        self.init = init
        self.with_bias = with_bias

        self.reset()

    def reset(self):
        if self.init == 'Xavier':
            w_bound = _np.sqrt(4. / (self.nin + self.nout))
            W = _np.random.uniform(low=-w_bound, high=w_bound,
                                   size=(self.nin, self.nout))
        else:
            raise NotImplementedError

        self.weight = _th.shared(W.astype(_th.config.floatX))
        self.grad_weight = _th.shared((W*0.).astype(_th.config.floatX))

        if self.with_bias:
            self.bias = _th.shared(_np.zeros(shape=self.nout, dtype=_th.config.floatX))
            self.grad_bias = _th.shared(_np.zeros(shape=self.nout, dtype=_th.config.floatX))

    def symb_forward(self, symb_input):
        out = _th.tensor.dot(symb_input, self.weight)

        if self.with_bias:
            out += self.bias

        return out
