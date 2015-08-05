from .Module import Module
from DeepFried2.init import const, xavier
from DeepFried2.utils import create_param_and_grad

import theano as _th
import numpy as _np


class SpatialConvolution(Module):
    def __init__(self, n_input_plane, n_output_plane, k_w, k_h, d_w=1, d_h=1, with_bias=True, initW=xavier(), initB=const(0), border_mode='valid', imshape=None):
        Module.__init__(self)
        self.n_input_plane = n_input_plane
        self.n_output_plane = n_output_plane
        self.k_w = k_w
        self.k_h = k_h
        self.d_w = d_w
        self.d_h = d_h
        self.with_bias = with_bias
        self.border_mode = border_mode
        self.imshape = imshape

        self.w_shape = (n_output_plane, n_input_plane, k_h, k_w)
        w_fan = (n_input_plane*k_w*k_h, n_output_plane*k_w*k_h)

        self.weight, self.grad_weight = create_param_and_grad(self.w_shape, initW, fan=w_fan, name='Wconv_{},{}@{}x{}'.format(n_input_plane, n_output_plane, k_w, k_h))
        if self.with_bias:
            self.bias, self.grad_bias = create_param_and_grad(n_output_plane, initB, name='bconv_{}'.format(n_output_plane))

    def symb_forward(self, symb_input):
        conv_output = _th.tensor.nnet.conv.conv2d(symb_input, self.weight,
            image_shape=(None, self.n_input_plane) + (self.imshape or (None, None)),
            filter_shape=self.w_shape,
            border_mode=self.border_mode,
            subsample=(self.d_h, self.d_w)
        )

        if self.with_bias:
            return conv_output + self.bias.dimshuffle('x', 0, 'x', 'x')
        else:
            return conv_output
