import DeepFried2 as df
from DeepFried2.utils import create_param_and_grad
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty
from theano.sandbox.cuda import dnn


class BackwardsConvolution3DCUDNN(df.Module):
    def __init__(self, n_input_vol, n_output_vol, k_w, k_h, k_d, d_w=1, d_h=1, d_d=1, pad_w=0, pad_h=0, pad_d=0, mode='cross', with_bias=True, initW=df.init.xavier(), initB=df.init.const(0), border='valid'):

        df.Module.__init__(self)
        self.n_input_vol = n_input_vol
        self.n_output_vol = n_output_vol
        self.k_w = k_w
        self.k_h = k_h
        self.k_d = k_d
        self.d_w = d_w
        self.d_h = d_h
        self.d_d = d_d
        self.mode = mode
        self.with_bias = with_bias

        # 'same' is a (common) shortcut for "zero-padding so that outshape == inshape".
        self.border = border or (pad_d, pad_h, pad_w)
        if self.border == 'same':
            assert self.k_w % 2 == 1 and self.k_h % 2 == 1 and self.k_d % 2 == 1, "'same' convolution only supports odd filter sizes."
            self.border = ((self.k_d-1)//2, (self.k_h-1)//2, (self.k_w-1)//2)

        w_shape = (n_input_vol, n_output_vol, k_d, k_h, k_w)
        w_fan = (n_output_vol*k_w*k_h*k_d, n_input_vol*k_w*k_h*k_d)

        self.weight, self.grad_weight = create_param_and_grad(w_shape, initW, fan=w_fan, name='Wconv_{},{},{},{},{}'.format(n_output_vol, n_input_vol, k_d, k_h, k_w))
        if self.with_bias:
            self.bias, self.grad_bias = create_param_and_grad((n_output_vol,), initB, name='bconv_{}'.format(n_output_vol))

    def symb_forward(self, symb_input):
        """symb_input shape: (n_input, channels, depth, height, width)"""

        if symb_input.ndim < 5:
            raise NotImplementedError('3D convolution requires a dimension >= 5')

        img = gpu_contiguous(symb_input)
        kerns = gpu_contiguous(self.weight)

        desc = dnn.GpuDnnConvDesc(border_mode=self.border, subsample=(self.d_d, self.d_h, self.d_w),
                              conv_mode=self.mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*self.d_d, img.shape[3]*self.d_h, img.shape[4]*self.d_w).shape, kerns.shape)
        out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*self.d_d, img.shape[3]*self.d_h, img.shape[4]*self.d_w)
        conv_output = dnn.GpuDnnConv3dGradI()(kerns, img, out, desc)

        if self.with_bias:
            return conv_output + self.bias.dimshuffle('x', 0, 'x', 'x', 'x')
        else:
            return conv_output
