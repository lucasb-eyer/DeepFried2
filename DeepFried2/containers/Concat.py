from .Container import Container

import theano.tensor as _T


class Concat(Container):
    def __init__(self, axis=1):
        Container.__init__(self)
        self.axis = axis

    def symb_forward(self, symb_input):
        symb_outputs = [module.symb_forward(symb_input) for module in self.modules]
        return _T.concatenate(symb_outputs, self.axis)
