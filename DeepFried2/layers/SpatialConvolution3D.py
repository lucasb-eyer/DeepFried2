import DeepFried2 as df
from DeepFried2.utils import create_param_and_grad

from theano.tensor.nnet import conv3d2d


class SpatialConvolution3D(df.Module):
    def __init__(self, n_input_vol, n_output_vol, k_w, k_h, k_d, with_bias=True, initW=df.init.xavier(), initB=df.init.const(0), border_mode='valid', volshape=None):
        df.Module.__init__(self)
        self.n_input_vol = n_input_vol
        self.n_output_vol = n_output_vol
        self.k_w = k_w
        self.k_h = k_h
        self.k_d = k_d
        self.with_bias = with_bias
        self.border_mode = border_mode
        self.volshape = volshape

        self.w_shape = (n_output_vol, k_d, n_input_vol, k_h, k_w)
        w_fan = (n_input_vol*k_w*k_h*k_d, n_output_vol*k_w*k_h*k_d)

        self.weight, self.grad_weight = create_param_and_grad(self.w_shape, initW, fan=w_fan, name='Wconv_{},{},{},{},{}'.format(n_output_vol, k_d, n_input_vol, k_h, k_w))
        if self.with_bias:
            self.bias, self.grad_bias = create_param_and_grad((n_output_vol,), initB, name='bconv_{}'.format(n_output_vol))

    def symb_forward(self, symb_input):

        """symb_input shape: (n_input, channels, depth, height, width)"""

        if symb_input.ndim < 5:
            raise NotImplementedError('3D convolution requires a dimension >= 5')

        # shuffle bcd01 -> bdc01
        symb_input = symb_input.swapaxes(1,2)

        conv_output = conv3d2d.conv3d(symb_input,
                                      self.weight,
                                      filters_shape=self.w_shape,
                                      border_mode=self.border_mode)

        # shuffle bdc01 -> bcd01
        conv_output = conv_output.swapaxes(1,2)

        if self.with_bias:
            return conv_output + self.bias.dimshuffle('x', 0, 'x', 'x', 'x')
        else:
            return conv_output
