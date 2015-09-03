import DeepFried2 as df


class AddConstant(df.Module):
    def __init__(self, scalar):
        df.Module.__init__(self)
        self.scalar = scalar

    def symb_forward(self, symb_input):
        return symb_input + self.scalar
