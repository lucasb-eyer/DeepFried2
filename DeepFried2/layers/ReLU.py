import DeepFried2 as df

from theano.tensor.nnet import relu

class ReLU(df.Module):

    def __init__(self, alpha=0, caxis=None):
        """ Fancy Rectified Linear Unit.
        - `alpha` is the "leakyness", i.e. slope of negative part (0=relu, 1=linear).
        - `caxis` can be specified to create a CReLU, [relu, -relu] along that axis.
        """
        df.Module.__init__(self)
        self.alpha = alpha
        self.caxis = caxis

    def symb_forward(self, x):
        if self.caxis is None:
            return relu(x, self.alpha)
        else:
            return df.T.concatenate([relu(x, self.alpha),
                                     relu(-x, self.alpha)], axis=self.caxis)
