import DeepFried2 as df
from DeepFried2.utils import create_param_and_grad
from theano.sandbox.cuda import dnn

import numpy as np

class SpatialConvolutionCUDNN(df.Module):
    def __init__(self, n_input_plane, n_output_plane, filter_size, stride=(1,1), padding=(0,0), mode='cross', with_bias=True, initW=df.init.xavier(), initB=df.init.const(0), border=None):
        # mode='cross' is the default in Lasagne[1], Torch[2], matConvNet[3], Caffee[4].
        #
        # 1: https://github.com/Lasagne/Lasagne/blob/63d44a0d/lasagne/layers/dnn.py#L299
        # 2: https://github.com/soumith/cudnn.torch/blob/840f0228/SpatialConvolution.lua#L83
        # 3: https://github.com/vlfeat/matconvnet/blob/b7dd9c96/matlab/src/bits/impl/nnconv_cudnn.cu#L133
        # 4: https://github.com/BVLC/caffe/blob/50ab52cb/include/caffe/util/cudnn.hpp#L104
        #
        # `border` is an alternative way to specify `pad_w` and `pad_h` so that Theano strings can be used. Better documentation to follow soon.
        df.Module.__init__(self)
        self.n_input_plane = n_input_plane
        self.n_output_plane = n_output_plane
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode
        self.with_bias = with_bias

        assert len(self.stride) == len(self.filter_size), 'The dimensionality of the stride and the filter size should match.'
        # 'same' is a (common) shortcut for "zero-padding so that outshape == inshape".
        self.border = border or padding
        if self.border == 'same':
            assert all(k % 2 == 1 for k in self.filter_size), "'same' convolution only supports odd filter sizes."

            self.border = tuple( (k - 1)//2 for k in self.filter_size )

        assert len(self.border) == len(self.stride), 'The dimensionality of the stride and the padding should match.'

        w_shape = (n_output_plane, n_input_plane) + self.filter_size
        w_fan = (np.prod(self.filter_size)*n_input_plane, np.prod(self.filter_size)*n_output_plane)

        param_name = 'Wconv_{},{}@{}' + 'x{}'*(len(w_shape) - 3)
        self.weight, self.grad_weight = create_param_and_grad(w_shape, initW, fan=w_fan, name=param_name.format(*w_shape))
        if self.with_bias:
            self.bias, self.grad_bias = create_param_and_grad(n_output_plane, initB, name='bconv_{}'.format(n_output_plane))

    def symb_forward(self, symb_input):
        conv = dnn.dnn_conv3d if symb_input.ndim == 5 else dnn.dnn_conv

        conv_output = conv(
            img=symb_input,
            kerns=self.weight,
            border_mode=self.border,
            subsample=self.stride,
            conv_mode=self.mode
        )

        if self.with_bias:
            d_shuffle = ('x', 0) + tuple('x') * (symb_input.ndim-2)
            return conv_output + self.bias.dimshuffle(*d_shuffle)
        else:
            return conv_output
