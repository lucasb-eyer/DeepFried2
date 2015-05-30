from .Module import Module

import theano as _th
import numpy as _np


class SpatialConvolution(Module):
    def __init__(self, n_input_plane, n_output_plane, k_w, k_h, d_w=1, d_h=1, with_bias=True, border_mode='valid', imshape=None):
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

        w_bound = _np.sqrt(4. / ((self.n_input_plane + self.n_output_plane) * self.k_w * self.k_h))
        W = _np.random.uniform(low=-w_bound, high=w_bound, size=(n_output_plane, n_input_plane, k_h, k_w))
        self.weight = _th.shared(W.astype(dtype=_th.config.floatX))
        self.grad_weight = _th.shared((W*0).astype(dtype=_th.config.floatX))

        if self.with_bias:
            self.bias = _th.shared(_np.zeros(shape=(n_output_plane, ), dtype=_th.config.floatX))
            self.grad_bias = _th.shared(_np.zeros(shape=(n_output_plane, ), dtype=_th.config.floatX))

    def symb_forward(self, symb_input):
        conv_output = _th.tensor.nnet.conv.conv2d(symb_input, self.weight,
            image_shape=(None, self.n_input_plane) + (self.imshape or (None, None)),
            filter_shape=(self.n_output_plane, self.n_input_plane, self.k_h, self.k_w),
            border_mode=self.border_mode,
            subsample=(self.d_h, self.d_w)
        )

        if self.with_bias:
            return conv_output + self.bias.dimshuffle('x', 0, 'x', 'x')
        else:
            return conv_output

