import DeepFried2 as df

_srng = df.th.sandbox.rng_mrg.MRG_RandomStreams()


class Dropout(df.Module):
    def __init__(self, dropout):
        df.Module.__init__(self)
        self.dropout = dropout

    def symb_forward(self, symb_input):
        if self.training_mode:
            shuffle_shape = (0, 1)
            if symb_input.ndim == 4:
                shuffle_shape += ('x', 'x')

            mask = _srng.binomial((symb_input.shape[0], symb_input.shape[1]),
                                  p=(1. - self.dropout),
                                  dtype='int32'
                                  ).astype(df.floatX).dimshuffle(*shuffle_shape)

            return symb_input / (1. - self.dropout) * mask
        else:
            return symb_input
