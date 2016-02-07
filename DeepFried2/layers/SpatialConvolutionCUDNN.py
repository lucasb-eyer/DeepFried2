import DeepFried2 as df
from DeepFried2.utils import expand
from theano.sandbox.cuda import dnn

import numpy as np

class SpatialConvolutionCUDNN(df.Module):
    def __init__(self, nchan_in, nchan_out, filter_size, stride=1, border=0, mode='cross', init=df.init.xavier(), bias=df.init.const(0)):
        # mode='cross' is the default in Lasagne[1], Torch[2], matConvNet[3], Caffee[4].
        #
        # 1: https://github.com/Lasagne/Lasagne/blob/63d44a0d/lasagne/layers/dnn.py#L299
        # 2: https://github.com/soumith/cudnn.torch/blob/840f0228/SpatialConvolution.lua#L83
        # 3: https://github.com/vlfeat/matconvnet/blob/b7dd9c96/matlab/src/bits/impl/nnconv_cudnn.cu#L133
        # 4: https://github.com/BVLC/caffe/blob/50ab52cb/include/caffe/util/cudnn.hpp#L104
        df.Module.__init__(self)
        self.nchan_in = nchan_in
        self.nchan_out = nchan_out
        self.filter_size = filter_size
        self.mode = mode
        self.stride = expand(stride, len(filter_size), 'stride')
        self.border = expand(border, len(filter_size), 'border')

        # 'same' is a (common) shortcut for "zero-padding so that outshape == inshape".
        if self.border == 'same':
            assert all(k % 2 == 1 for k in self.filter_size), "'same' convolution only supports odd filter sizes."
            self.border = tuple( (k - 1)//2 for k in self.filter_size )

        w_shape = (nchan_out, nchan_in) + self.filter_size
        w_fan = (np.prod(self.filter_size)*nchan_in, np.prod(self.filter_size)*nchan_out)
        w_name = ('Wconv_{},{}@{}' + 'x{}'*(len(w_shape) - 3)).format(*w_shape)
        self.W = self._addparam(w_shape, init, fan=w_fan, name=w_name)

        if bias not in (None, False):
            self.b = self._addparam(nchan_out, bias, decay=False, name='bconv_{}'.format(nchan_out))
        else:
            self.b = None


    def symb_forward(self, symb_input):
        # Check for filter-size instead of input dims, as input dimension 5 could
        # mean a time-series of images just as well as a 3D volume.
        conv = dnn.dnn_conv3d if len(self.filter_size) == 3 else dnn.dnn_conv

        conv_output = conv(
            img=symb_input,
            kerns=self.W.param,
            border_mode=self.border,
            subsample=self.stride,
            conv_mode=self.mode
        )

        if self.b is not None:
            d_shuffle = ('x', 0) + tuple('x') * (symb_input.ndim-2)
            conv_output += self.b.param.dimshuffle(*d_shuffle)

        return conv_output
