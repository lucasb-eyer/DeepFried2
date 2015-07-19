from .Module import Module
from beacon8.init import zero, xavier
from beacon8.utils import create_param_and_grad

import theano as _th
import numpy as _np
import theano.sandbox.cuda.dnn as _dnn


class SpatialConvolutionCUDNN(Module):
    def __init__(self, n_input_plane, n_output_plane, k_w, k_h, d_w=1, d_h=1, pad_w=0, pad_h=0, with_bias=True, init=xavier, init_b=zero):
        Module.__init__(self)
        self.n_input_plane = n_input_plane
        self.n_output_plane = n_output_plane
        self.k_w = k_w
        self.k_h = k_h
        self.d_w = d_w
        self.d_h = d_h
        self.pad_w = pad_w
        self.pad_h = pad_h
        self.with_bias = with_bias

        w_shape = (n_output_plane, n_input_plane, k_h, k_w)
        w_fan = (n_input_plane*k_w*k_h, n_output_plane*k_w*k_h)

        self.weight, self.grad_weight = create_param_and_grad(w_shape, init, fan=w_fan, name='Wconv_{},{}@{}x{}'.format(n_input_plane, n_output_plane, k_w, k_h))
        if self.with_bias:
            self.bias, self.grad_bias = create_param_and_grad(n_output_plane, init_b, name='bconv_{}'.format(n_output_plane))

    def symb_forward(self, symb_input):
        conv_output = _dnn.dnn_conv(img=symb_input,
                                    kerns=self.weight,
                                    border_mode=(self.pad_h, self.pad_w),
                                    subsample=(self.d_h, self.d_w))

        if self.with_bias:
            return conv_output + self.bias.dimshuffle('x', 0, 'x', 'x')
        else:
            return conv_output
