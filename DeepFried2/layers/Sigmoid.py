from .Module import Module

import theano.tensor as _T


class Sigmoid(Module):
    def __init__(self, fn=_T.nnet.sigmoid):
        Module.__init__(self)
        self.fn = fn

    @classmethod
    def ultrafast(cls):
        return cls(_T.nnet.ultra_fast_sigmoid)

    @classmethod
    def hard(cls):
        return cls(_T.nnet.hard_sigmoid)

    def symb_forward(self, symb_input):
        return self.fn(symb_input)
