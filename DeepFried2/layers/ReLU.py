import DeepFried2 as df

from theano.tensor.nnet import relu

class ReLU(df.Module):

    def __init__(self, alpha = 0):
        df.Module.__init__(self)
        self.alpha = alpha

    def symb_forward(self, symb_input):
        return relu(symb_input, self.alpha)
