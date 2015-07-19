from .Module import Module
from beacon8.init import zero, xavier
from beacon8.utils import create_param_and_grad

import numpy as _np
import theano as _th


class Linear(Module):

    def __init__(self, nin, nout, init=xavier, with_bias=True, init_b=zero):
        Module.__init__(self)

        self.nin = nin
        self.nout = nout
        self.with_bias = with_bias

        self.weight, self.grad_weight = create_param_and_grad((nin, nout), init, fan=(nin, nout), name='Wlin_{}x{}'.format(nin, nout))
        if self.with_bias:
            self.bias, self.grad_bias = create_param_and_grad(nout, init_b, name='blin_{}'.format(nout))

    def symb_forward(self, symb_input):
        out = _th.tensor.dot(symb_input, self.weight)

        if self.with_bias:
            out += self.bias

        return out

