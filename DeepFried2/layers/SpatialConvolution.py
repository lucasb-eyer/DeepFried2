import DeepFried2 as df
from DeepFried2.utils import expand
import numpy as np
from theano.tensor.nnet import conv3d2d

class SpatialConvolution(df.Module):
    def __init__(self, nchan_in, nchan_out, filter_size, stride=1, border='valid', mode='cross', init=df.init.xavier(), bias=0, imshape=None):
        # See `SpatialConvolutionCUDNN` comment for the `mode` parameter. Only works in 2D
        df.Module.__init__(self)
        self.nchan_in = nchan_in
        self.nchan_out = nchan_out
        self.filter_size = filter_size
        self.mode = mode
        self.imshape = expand(imshape, len(filter_size), 'imshape', expand_nonnum=True)
        self.stride = expand(stride, len(filter_size), 'stride')
        self.border = expand(border, len(filter_size), 'border')

        if len(self.filter_size) == 3 and any(s != 1 for s in self.stride):
            raise NotImplementedError('stride != 1 is not implemented for 3D convolutions')

        if len(self.filter_size) == 3 and imshape is not None:
            raise NotImplementedError('imshape is not implemented for 3D convolutions')

        if len(self.filter_size) == 3 and mode != 'conv':
            raise NotImplementedError('mode="cross" is not implemented for 3D convolutions')

        if mode not in ('conv', 'cross'):
            raise NotImplementedError('Only "conv" and "cross" modes are implemented')

        self.w_shape = (nchan_out, nchan_in) + self.filter_size
        w_fan = (nchan_in*np.prod(self.filter_size), nchan_out*np.prod(self.filter_size))
        w_name = ('Wconv_{},{}@{}' + 'x{}'*(len(self.w_shape) - 3)).format(*self.w_shape)
        self.W = self._addparam(self.w_shape, init, fan=w_fan, name=w_name)
        self.b = self._addparam_optional(nchan_out, bias, decay=False, name='bconv_{}'.format(nchan_out))


    def symb_forward(self, symb_input):
        mode = self.border
        input_shape = symb_input.shape

        if len(self.filter_size) == 3:
        # Implement 'same' convolution by padding upfront. (TODO: use theano's 'half'? Is it supported in 3d?)
            if mode == 'same':
                if any(d != 1 for d in self.stride):
                    raise NotImplementedError("'same' is not implement for strides != 1 (try CUDNN)")
                mode = 'valid'
                symb_input = df.utils.pad(symb_input, (0,0) + tuple( (k - 1)//2 for k in self.filter_size ))
            # 'full' is not implemented in 3D, so work-around by padding upfront.
            # 3D is forced to use 'valid', so we're not setting anything for that here.
            elif mode == 'full':
                symb_input = df.utils.pad(symb_input, (0,0) + tuple( k - 1 for k in self.filter_size ))
            # If a specific padding is set, convolution is always "normal", i.e. 'valid'.
            elif isinstance(mode, tuple):
                mode = 'valid'
                symb_input = df.utils.pad(symb_input, (0,0) + self.border)

            # shuffle bcd01 -> bdc01
            conv_output = df.T.nnet.conv3d2d.conv3d(symb_input.swapaxes(1,2),
                    self.W.param.swapaxes(1,2),
                    border_mode='valid'
            )
            # shuffle bdc01 -> bcd01
            conv_output = conv_output.swapaxes(1,2)
        else:
            if mode == 'same':
                mode = 'half'

            conv_output = df.T.nnet.conv2d(symb_input, self.W.param,
                input_shape=(None, self.nchan_in) + self.imshape,
                filter_shape=self.w_shape,
                border_mode=mode,
                subsample=self.stride,
                filter_flip={'conv': True, 'cross': False}[self.mode],
            )

        if self.b is not None:
            d_shuffle = ('x', 0) + ('x',) * (symb_input.ndim-2)
            conv_output += self.b.param.dimshuffle(*d_shuffle)

        return conv_output
