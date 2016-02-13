import DeepFried2 as df

from theano.tensor.nnet import elu

class ELU(df.Module):

    def __init__(self, alpha = 1):
        df.Module.__init__(self)
        self.alpha = alpha

    def symb_forward(self, symb_input):
        return elu(symb_input, self.alpha)
