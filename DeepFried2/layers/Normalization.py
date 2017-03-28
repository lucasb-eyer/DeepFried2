import DeepFried2 as df

class Normalization(df.Module):
    def __init__(self, axis=-1, eps=1e-8):
        df.Module.__init__(self)
        self.axis = axis
        self.eps = eps

    def symb_forward(self, symb_in):
        return symb_in / df.T.sqrt(self.eps + (symb_in**2).sum(axis=self.axis, keepdims=True))
