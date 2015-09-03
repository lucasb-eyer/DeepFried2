import DeepFried2 as df


class Sigmoid(df.Module):
    def __init__(self, fn=df.T.nnet.sigmoid):
        df.Module.__init__(self)
        self.fn = fn

    @classmethod
    def ultrafast(cls):
        return cls(df.T.nnet.ultra_fast_sigmoid)

    @classmethod
    def hard(cls):
        return cls(df.T.nnet.hard_sigmoid)

    def symb_forward(self, symb_input):
        return self.fn(symb_input)
