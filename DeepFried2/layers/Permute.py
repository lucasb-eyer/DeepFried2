import DeepFried2 as df


class Permute(df.Module):
    def __init__(self, *new_shape):
        df.Module.__init__(self)
        self.new_shape = new_shape

    def symb_forward(self, symb_input):
        return symb_input.dimshuffle(*self.new_shape)
