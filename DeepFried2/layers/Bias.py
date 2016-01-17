import DeepFried2 as df
from DeepFried2.utils import create_param_and_grad, aslist


class Bias(df.Module):

    def __init__(self, shape, init=df.init.const(0), bcast=None):
        df.Module.__init__(self)

        self.bias, self.grad_bias = create_param_and_grad(shape, init, name='b_{}'.format(shape), broadcastable=bcast)

    def symb_forward(self, symb_input):
        return symb_input + self.bias
