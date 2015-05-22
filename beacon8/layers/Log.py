from .Module import Module

import theano.tensor as _T


class Log(Module):
    def symb_forward(self, symb_input):
        return _T.log(symb_input)
