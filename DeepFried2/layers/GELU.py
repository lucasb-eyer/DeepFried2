import DeepFried2 as df

class GELU(df.Module):
    """ Gaussian Error Linear Unit (https://arxiv.org/abs/1606.08415) """

    def symb_forward(self, x):
        """ A very close, much more efficient approximation. """
        return 0.5 * x * (1 + df.T.tanh(0.79788456 * (x + 0.044715 * x*x*x)))

