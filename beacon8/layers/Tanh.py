from .Module import Module

import theano.tensor as _T


class Tanh(Module):

    def symb_forward(self, symb_input):
        return _T.tanh(symb_input)
