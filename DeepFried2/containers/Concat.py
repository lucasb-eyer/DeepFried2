from .Container import Container

import theano.tensor as _T


class Concat(Container):
    def __init__(self, axis=1):
        Container.__init__(self)
        self.axis = axis

    def symb_forward(self, symb_inputs):
        assert isinstance(symb_inputs, (list, tuple)), "Input to `Concat` container needs to be a tuple or a list."
        return _T.concatenate(symb_inputs, self.axis)
