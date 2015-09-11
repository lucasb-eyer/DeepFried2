import DeepFried2 as df

import theano.tensor as _T


class SpatialMaxPooling3D(df.Module):
    def __init__(self, k_w, k_h, k_d, d_w=None, d_h=None, d_d=None, ignore_border=False):
        df.Module.__init__(self)
        self.k_w = k_w
        self.k_h = k_h
        self.k_d = k_d
        self.ignore_border = ignore_border

        if d_w is None:
            self.d_w = self.k_w
        else:
            self.d_w = d_w

        if d_h is None:
            self.d_h = self.k_h
        else:
            self.d_h = d_h

        if d_d is None:
            self.d_d = self.k_d
        else:
            self.d_d = d_d

    def symb_forward(self, symb_input):
        """ 3d max pooling taken from github.com/lpigou/Theano-3D-ConvNet/
            (with modified shuffeling) """
        if symb_input.ndim < 5:
            raise NotImplementedError('max pooling 3D requires a dimension >= 5')

        height_width_shape = symb_input.shape[-2:]

        batch_size = _T.prod(symb_input.shape[:-2])
        batch_size = _T.shape_padright(batch_size, 1)

        new_shape = _T.cast(_T.join(0, batch_size, _T.as_tensor([1,]), height_width_shape), 'int32')

        input_4d = _T.reshape(symb_input, new_shape, ndim=4)

        # downsample height and width first
        # other dimensions contribute to batch_size
        op = _T.signal.downsample.DownsampleFactorMax((self.k_h, self.k_w), self.ignore_border, st=(self.d_h, self.d_w))
        output = op(input_4d)

        outshape = _T.join(0, symb_input.shape[:-2], output.shape[-2:])
        out = _T.reshape(output, outshape, ndim=symb_input.ndim)

        vol_dim = symb_input.ndim

        shufl = (list(range(vol_dim-4)) + [vol_dim-2]+[vol_dim-1]+[vol_dim-3]+[vol_dim-4])
        input_depth = out.dimshuffle(shufl)
        vol_shape = input_depth.shape[-2:]

        batch_size = _T.prod(input_depth.shape[:-2])
        batch_size = _T.shape_padright(batch_size,1)

        new_shape = _T.cast(_T.join(0, batch_size, _T.as_tensor([1,]), vol_shape), 'int32')
        input_4D_depth = _T.reshape(input_depth, new_shape, ndim=4)

        # downsample depth
        # other dimensions contribute to batch_size
        op = _T.signal.downsample.DownsampleFactorMax((1,self.k_d), self.ignore_border, st=(1,self.d_d))
        outdepth = op(input_4D_depth)

        outshape = _T.join(0, input_depth.shape[:-2], outdepth.shape[-2:])
        shufl = (list(range(vol_dim-4)) + [vol_dim-1]+[vol_dim-2]+[vol_dim-4]+[vol_dim-3])

        return _T.reshape(outdepth, outshape, ndim=symb_input.ndim).dimshuffle(shufl)
