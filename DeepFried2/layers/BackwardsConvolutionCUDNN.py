import DeepFried2 as df
from DeepFried2.utils import create_param_and_grad
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty
from theano.sandbox.cuda import dnn

import numpy as np


class BackwardsConvolutionCUDNN(df.Module):
    def __init__(self, n_input_plane, n_output_plane, filter_size, stride=(1,1), padding=(0,0), mode='cross', with_bias=True, initW=df.init.xavier(), initB=df.init.const(0), border=None):
        # mode='cross' is the default in Lasagne[1], Torch[2], matConvNet[3], Caffee[4].
        #
        # 1: https://github.com/Lasagne/Lasagne/blob/63d44a0d/lasagne/layers/dnn.py#L299
        # 2: https://github.com/soumith/cudnn.torch/blob/840f0228/SpatialConvolution.lua#L83
        # 3: https://github.com/vlfeat/matconvnet/blob/b7dd9c96/matlab/src/bits/impl/nnconv_cudnn.cu#L133
        # 4: https://github.com/BVLC/caffe/blob/50ab52cb/include/caffe/util/cudnn.hpp#L104

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

        w_shape = (n_input_plane, n_output_plane) + self.filter_size
        w_fan = (np.prod(self.filter_size)*n_output_plane, np.prod(self.filter_size)*n_input_plane)

        param_name = 'Wconv_{},{}@{}' + 'x{}'*(len(w_shape) - 3)
        self.weight, self.grad_weight = create_param_and_grad(w_shape, initW, fan=w_fan, name=param_name.format(*w_shape))
        if self.with_bias:
            self.bias, self.grad_bias = create_param_and_grad(n_output_plane, initB, name='bconv_{}'.format(n_output_plane))


    def symb_forward(self, symb_input):
        """ creates dummy forward conv and uses its gradient as backwards pass """
        """ This code is mostly taken from https://github.com/Newmu/dcgan_code/blob/master/lib/ops.py """
        img = gpu_contiguous(symb_input)
        kerns = gpu_contiguous(self.weight)

        alloc_shape = (img.shape[0], kerns.shape[1]) + tuple(i*d for i,d in zip(img.shape[2:],self.stride))
        desc = dnn.GpuDnnConvDesc(border_mode=self.border, subsample=self.stride, conv_mode=self.mode)(gpu_alloc_empty(*alloc_shape).shape, kerns.shape)
        out = gpu_alloc_empty(*alloc_shape)
        grad = dnn.GpuDnnConv3dGradI if symb_input.ndim == 5 else dnn.GpuDnnConvGradI
        conv_output = grad()(kerns, img, out, desc)

        if self.with_bias:
            d_shuffle = ('x', 0) + tuple('x') * (symb_input.ndim-2)
            return conv_output + self.bias.dimshuffle(*d_shuffle)
        else:
            return conv_output
