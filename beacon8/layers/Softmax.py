from . import Module

import theano.tensor as _T


class SoftMax(Module):

    def __init__(self):
        Module.__init__(self)

    def symb_forward(self, symb_input):
        return _T.nnet.softmax(symb_input)
