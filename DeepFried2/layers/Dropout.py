import DeepFried2 as df

_srng = df.th.sandbox.rng_mrg.MRG_RandomStreams()


class Dropout(df.Module):
    def __init__(self, dropout):
        df.Module.__init__(self)
        self.dropout = dropout

    def symb_forward(self, symb_input):
        if self.training_mode:
            mask = _srng.binomial(symb_input.shape,
                                  p=(1. - self.dropout),
                                  dtype=df.floatX
                                  )

            return symb_input / (1. - self.dropout) * mask
        else:
            return symb_input
