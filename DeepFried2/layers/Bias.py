import DeepFried2 as df


class Bias(df.Module):

    def __init__(self, shape, init=df.init.const(0), bcast=None):
        df.Module.__init__(self)
        self.b = self._addparam(shape, init, name='b_{}'.format(shape), broadcastable=bcast, decay=False)

    def symb_forward(self, symb_input):
        return symb_input + self.b.param
