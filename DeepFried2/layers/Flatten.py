import DeepFried2 as df


class Flatten(df.Module):
    def __init__(self, ndim=2):
        df.Module.__init__(self)
        self.ndim = ndim

    def symb_forward(self, symb_input):
        return symb_input.flatten(self.ndim)

