import DeepFried2 as df
from DeepFried2.utils import create_param_and_grad


class SpatialConvolution(df.Module):
    def __init__(self, n_input_plane, n_output_plane, k_w, k_h, d_w=1, d_h=1, pad_w=0, pad_h=0, with_bias=True, initW=df.init.xavier(), initB=df.init.const(0), border_mode='valid', imshape=None):
        df.Module.__init__(self)
        self.n_input_plane = n_input_plane
        self.n_output_plane = n_output_plane
        self.k_w = k_w
        self.k_h = k_h
        self.d_w = d_w
        self.d_h = d_h
        self.with_bias = with_bias
        self.border_mode = border_mode or (pad_h, pad_w)
        self.imshape = imshape

        self.w_shape = (n_output_plane, n_input_plane, k_h, k_w)
        w_fan = (n_input_plane*k_w*k_h, n_output_plane*k_w*k_h)

        self.weight, self.grad_weight = create_param_and_grad(self.w_shape, initW, fan=w_fan, name='Wconv_{},{}@{}x{}'.format(n_input_plane, n_output_plane, k_w, k_h))
        if self.with_bias:
            self.bias, self.grad_bias = create_param_and_grad(n_output_plane, initB, name='bconv_{}'.format(n_output_plane))

    def symb_forward(self, symb_input):
        mode = self.border_mode
        input_shape = symb_input.shape

        if self.border_mode == 'same':
            if self.d_w != 1 or self.d_h != 1:
                raise NotImplementedError("'same' is not implement for strides != 1")
            mode = 'full'
            pad_h = (self.k_h - 1) // 2
            pad_w = (self.k_w - 1) // 2
        elif isinstance(self.border_mode, tuple):
            mode = 'valid'
            pad_h, pad_w = self.border_mode
        else:
            pad_h = 0
            pad_w = 0

        if pad_h != 0 or pad_w != 0:
            symb_input = df.utils.pad(symb_input, (0,0,pad_h,pad_w))

        conv_output = df.T.nnet.conv.conv2d(symb_input, self.weight,
                                            image_shape=(None, self.n_input_plane) + (self.imshape or (None, None)),
                                            filter_shape=self.w_shape,
                                            border_mode=mode,
                                            subsample=(self.d_h, self.d_w)
        )

        if self.border_mode == 'same':
            conv_output = conv_output[:,:,pad_h:input_shape[2]+pad_h,pad_w:input_shape[3]+pad_w]


        if self.with_bias:
            return conv_output + self.bias.dimshuffle('x', 0, 'x', 'x')
        else:
            return conv_output
