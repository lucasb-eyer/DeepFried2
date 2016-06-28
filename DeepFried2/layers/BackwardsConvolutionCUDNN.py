import DeepFried2 as df
from DeepFried2.utils import expand
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty
from theano.sandbox.cuda import dnn

import numpy as np


class BackwardsConvolutionCUDNN(df.Module):
    def __init__(self, nchan_in, nchan_out, filter_size, stride=1, border=0, mode='cross', init=df.init.xavier(), bias=df.init.const(0)):
        """
        This is the backwards path through a convolution, sometimes is also
        referred to as transposed convolution and (wrongly) deconvolution.

        This is usually used for upsampling an image. If you want the exact
        counterpart to another convolution earlier part of your model, consider
        using the `backward` function with that convolution instead.

        - `nchan_in`: number of channels in the input.
        - `nchan_out`: number of filters and thus channels in the output.
        - `filter_size`: 2D or 3D tuple describing the filter size.
        - `stride`: the stride "dilates" the output, i.e. makes it larger.
        - `border`: The counterpart to `border` in forward convolution. This
            effectively crops the output, as opposed to padding it.
        - `mode`: `'cross'` or `'conv'`, see forward convolution documentation.
        - `init`: initializer for the weights/filters.
        - `bias`: initializer for the bias, or `None` or `False`.
        """
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

        w_shape = (nchan_in, nchan_out) + self.filter_size
        w_fan = (np.prod(self.filter_size)*nchan_out, np.prod(self.filter_size)*nchan_in)
        w_name = ('Wconv_{},{}@{}' + 'x{}'*(len(w_shape) - 3)).format(*w_shape)
        self.W = self._addparam(w_shape, init, fan=w_fan, name=w_name)

        if bias not in (None, False):
            self.b = self._addparam(nchan_out, bias, decay=False, name='bconv_{}'.format(nchan_out))
        else:
            self.b = None


    def symb_forward(self, symb_input):
        # Calls directly into CUDNN's gradient methods to insert a backward-conv Op.
        # This code is originally taken from https://github.com/Newmu/dcgan_code/blob/master/lib/ops.py
        # and extended to more complex scenarios (stride, border)
        img = gpu_contiguous(symb_input)
        kerns = gpu_contiguous(self.W.param)

        alloc_shape = (img.shape[0], self.nchan_out) + tuple((i-1)*s - 2*b + f for i,s,b,f in zip(img.shape[2:], self.stride, self.border, self.filter_size))
        out = gpu_alloc_empty(*alloc_shape)
        desc = dnn.GpuDnnConvDesc(border_mode=self.border, subsample=self.stride, conv_mode=self.mode)(out.shape, kerns.shape)
        grad = dnn.GpuDnnConv3dGradI if symb_input.ndim == 5 else dnn.GpuDnnConvGradI
        conv_output = grad()(kerns, img, out, desc)

        if self.b is not None:
            d_shuffle = ('x', 0) + tuple('x') * (symb_input.ndim-2)
            conv_output += self.b.param.dimshuffle(*d_shuffle)

        return conv_output
