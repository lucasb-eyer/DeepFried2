from . import Module

import theano.tensor as _T

class SoftMax(Module):

    def __init__(self):
        Module.__init__(self)

    def symbolic_forward(self, symbolic_input):
        return _T.nnet.softmax(symbolic_input)
