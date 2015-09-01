from .Module import Module
from DeepFried2.init import const, xavier
from DeepFried2.utils import create_param_and_grad

import theano as _th
from theano.tensor.nnet import conv3d2d
import numpy as _np


class SpatialConvolution3D(Module):
    def __init__(self, n_input_vol, n_output_vol, k_w, k_h, k_d, with_bias=True, initW=xavier(), initB=const(0), border_mode='valid', volshape=None):
        Module.__init__(self)
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

        """symb_input shape: (n_input, depth, channels, height, width)"""

        conv_output = conv3d2d.conv3d(symb_input,
                                      self.weight,
                                      filters_shape=self.w_shape,
                                      border_mode=self.border_mode)

        if self.with_bias:
            return conv_output + self.bias.dimshuffle('x', 'x', 0, 'x', 'x')
        else:
            return conv_output
