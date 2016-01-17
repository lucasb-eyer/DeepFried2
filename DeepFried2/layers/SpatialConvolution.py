import DeepFried2 as df
from DeepFried2.utils import create_param_and_grad
import numpy as np

from theano.tensor.nnet import conv3d2d

class SpatialConvolution(df.Module):
    def __init__(self, n_input_plane, n_output_plane, filter_size, stride=(1,1), padding=(0,0), with_bias=True, initW=df.init.xavier(), initB=df.init.const(0), border_mode='valid', imshape=None):
        df.Module.__init__(self)
        self.n_input_plane = n_input_plane
        self.n_output_plane = n_output_plane
        self.filter_size = filter_size
        self.stride = stride
        self.with_bias = with_bias
        self.border_mode = border_mode or padding
        self.imshape = imshape

        assert len(self.stride) == len(self.filter_size), 'The dimensionality of the stride and the filter size should match.'

        if len(self.filter_size) == 3 and any(s != 1 for s in stride):
            raise NotImplementedError('stride != 1 is not implemented for 3D convolutions')

        if len(self.filter_size) == 3 and imshape is not None:
            raise NotImplementedError('imshape is not implemented for 3D convolutions')

        self.w_shape = (n_output_plane, n_input_plane) + self.filter_size
        w_fan = (n_input_plane*np.prod(self.filter_size), n_output_plane*np.prod(self.filter_size))

        param_name = 'Wconv_{},{}@{}' + 'x{}'*(len(self.w_shape) - 3)
        self.weight, self.grad_weight = create_param_and_grad(self.w_shape, initW, fan=w_fan, name=param_name.format(*self.w_shape))
        if self.with_bias:
            self.bias, self.grad_bias = create_param_and_grad(n_output_plane, initB, name='bconv_{}'.format(n_output_plane))

    def symb_forward(self, symb_input):
        mode = self.border_mode
        input_shape = symb_input.shape

        if self.border_mode == 'same':
            if any(d != 1 for d in self.stride):
                raise NotImplementedError("'same' is not implement for strides != 1")
            mode = 'full'
            padding = tuple( (k - 1)//2 for k in self.filter_size )
        elif self.border_mode == 'full' and symb_input.ndim == 5:
            padding = tuple( k - 1 for k in self.filter_size )
        elif isinstance(self.border_mode, tuple):
            mode = 'valid'
            padding = self.border_mode
        else:
            padding = tuple([0] * len(self.filter_size))

        if any(p != 0 for p in padding):
            symb_input = df.utils.pad(symb_input, (0,0) + padding)

        if symb_input.ndim == 5:
            # shuffle bcd01 -> bdc01
            conv_output = conv3d2d.conv3d(symb_input.swapaxes(1,2),
                    self.weight.swapaxes(1,2),
                    border_mode='valid')
            # shuffle bdc01 -> bcd01
            conv_output = conv_output.swapaxes(1,2)
        else:
            conv_output = df.T.nnet.conv.conv2d(symb_input, self.weight,
                    image_shape=(None, self.n_input_plane) + (self.imshape or (None, None)),
                    filter_shape=self.w_shape,
                    border_mode=mode,
                    subsample=self.stride
                    )

            if self.border_mode == 'same':
                conv_output = conv_output[:,:,padding[0]:input_shape[2]+padding[0],padding[1]:input_shape[3]+padding[1]]


        if self.with_bias:
            d_shuffle = ('x', 0) + tuple('x') * (symb_input.ndim-2)
            return conv_output + self.bias.dimshuffle(*d_shuffle)
        else:
            return conv_output
