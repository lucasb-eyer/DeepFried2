import DeepFried2 as df
from theano.sandbox.rng_mrg import MRG_RandomStreams

_srng = MRG_RandomStreams()


class Dropout(df.Module):
    def __init__(self, dropout):
        df.Module.__init__(self)
        self.dropout = dropout

    def symb_forward(self, symb_input):
        if self._mode == 'train':
            mask = _srng.binomial(symb_input.shape,
                                  p=(1. - self.dropout),
                                  dtype=df.floatX
                                  )

            return symb_input / (1. - self.dropout) * mask
        else:
            return symb_input
