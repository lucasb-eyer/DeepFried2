import DeepFried2 as df

import numpy as _np


class Linear(df.Module):

    def __init__(self, nin, nout, init=df.init.xavier(), bias=df.init.const(0)):
        df.Module.__init__(self)

        self.nin = nin
        self.nout = nout

        shape = (nin, nout)
        self.W = self._addparam(shape, init, fan=shape, name='Wlin_{}x{}'.format(*shape))
        if bias not in (None, False):
            self.b = self._addparam(nout, bias, decay=False, name='blin_{}'.format(nout))
        else:
            self.b = None

    def symb_forward(self, symb_input):
        out = df.T.dot(symb_input, self.W.param)

        if self.b is not None:
            out += self.b.param

        return out
