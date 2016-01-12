import DeepFried2 as df
from DeepFried2.utils import create_param_and_grad
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty
from theano.sandbox.cuda import dnn


class BackwardsConvolutionCUDNN(df.Module):
    def __init__(self, n_input_plane, n_output_plane, k_w, k_h, d_w=1, d_h=1, pad_w=0, pad_h=0, mode='cross', with_bias=True, initW=df.init.xavier(), initB=df.init.const(0), border=None):
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
        self.k_w = k_w
        self.k_h = k_h
        self.d_w = d_w
        self.d_h = d_h
        self.mode = mode
        self.with_bias = with_bias

        # 'same' is a (common) shortcut for "zero-padding so that outshape == inshape".
        self.border = border or (pad_h, pad_w)
        if self.border == 'same':
            assert self.k_w % 2 == 1 and self.k_h % 2 == 1, "'same' convolution only supports odd filter sizes."
            self.border = ((self.k_h-1)//2, (self.k_w-1)//2)

        w_shape = (n_input_plane, n_output_plane, k_h, k_w)
        w_fan = (n_output_plane*k_w*k_h, n_input_plane*k_w*k_h)

        self.weight, self.grad_weight = create_param_and_grad(w_shape, initW, fan=w_fan, name='Wconv_{},{}@{}x{}'.format(n_input_plane, n_output_plane, k_w, k_h))
        if self.with_bias:
            self.bias, self.grad_bias = create_param_and_grad(n_output_plane, initB, name='bconv_{}'.format(n_output_plane))

    def symb_forward(self, symb_input):
        """ creates dummy forward conv and uses its gradient as backwards pass """
        img = gpu_contiguous(symb_input)
        kerns = gpu_contiguous(self.weight)
        desc = dnn.GpuDnnConvDesc(border_mode=self.border, subsample=(self.d_h, self.d_w),
                conv_mode=self.mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*self.d_h, img.shape[3]*self.d_w).shape, kerns.shape)
        out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*self.d_h, img.shape[3]*self.d_w)
        conv_output = dnn.GpuDnnConvGradI()(kerns, img, out, desc)

        if self.with_bias:
            return conv_output + self.bias.dimshuffle('x', 0, 'x', 'x')
        else:
            return conv_output
