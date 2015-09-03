import DeepFried2 as df
from DeepFried2.utils import create_param_and_grad

import numpy as _np


class Linear(df.Module):

    def __init__(self, nin, nout, with_bias=True, initW=df.init.xavier(), initB=df.init.const(0)):
        df.Module.__init__(self)

        self.nin = nin
        self.nout = nout
        self.with_bias = with_bias

        self.weight, self.grad_weight = create_param_and_grad((nin, nout), initW, fan=(nin, nout), name='Wlin_{}x{}'.format(nin, nout))
        if self.with_bias:
            self.bias, self.grad_bias = create_param_and_grad(nout, initB, name='blin_{}'.format(nout))

    def symb_forward(self, symb_input):
        out = df.T.dot(symb_input, self.weight)

        if self.with_bias:
            out += self.bias

        return out

