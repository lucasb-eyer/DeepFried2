import DeepFried2 as df
from theano.tensor.signal.pool import pool_2d, Pool


class SpatialMaxPooling(df.Module):
    def __init__(self, window_size, stride=None, padding=None, ignore_border=True):
        df.Module.__init__(self)
        self.window_size = window_size
        self.ignore_border = ignore_border

        if stride is None:
            self.stride = window_size
        else:
            self.stride = stride

        if padding is None:
            self.padding = (0,) * len(self.window_size)
        else:
            self.padding = padding


    def symb_forward(self, symb_input):
        if symb_input.ndim == 5:
            """ 3d max pooling taken from github.com/lpigou/Theano-3D-ConvNet/
            (with modified shuffeling)
            symb_input shape: (n_input, channels, depth, height, width)"""

            height_width_shape = symb_input.shape[-2:]

            batch_size = df.T.prod(symb_input.shape[:-2])
            batch_size = df.T.shape_padright(batch_size, 1)

            new_shape = df.T.cast(df.T.join(0, batch_size, df.T.as_tensor([1,]), height_width_shape), 'int32')

            input_4d = df.T.reshape(symb_input, new_shape, ndim=4)

            # downsample height and width first
            # other dimensions contribute to batch_size
            op = Pool(self.window_size[1:], self.ignore_border, st=self.stride[1:], padding=self.padding[1:])
            output = op(input_4d)

            outshape = df.T.join(0, symb_input.shape[:-2], output.shape[-2:])
            out = df.T.reshape(output, outshape, ndim=symb_input.ndim)

            vol_dim = symb_input.ndim

            shufl = (list(range(vol_dim-4)) + [vol_dim-2]+[vol_dim-1]+[vol_dim-4]+[vol_dim-3])
            input_depth = out.dimshuffle(shufl)
            vol_shape = input_depth.shape[-2:]

            batch_size = df.T.prod(input_depth.shape[:-2])
            batch_size = df.T.shape_padright(batch_size,1)

            new_shape = df.T.cast(df.T.join(0, batch_size, df.T.as_tensor([1,]), vol_shape), 'int32')
            input_4D_depth = df.T.reshape(input_depth, new_shape, ndim=4)

            # downsample depth
            # other dimensions contribute to batch_size
            op = Pool((1,self.window_size[0]), self.ignore_border, st=(1,self.stride[0]), padding=(0, self.padding[0]))
            outdepth = op(input_4D_depth)

            outshape = df.T.join(0, input_depth.shape[:-2], outdepth.shape[-2:])
            shufl = (list(range(vol_dim-4)) + [vol_dim-2]+[vol_dim-1]+[vol_dim-4]+[vol_dim-3])

            return df.T.reshape(outdepth, outshape, ndim=symb_input.ndim).dimshuffle(shufl)
        else:
            return pool_2d(
                symb_input,
                ds=self.window_size,
                ignore_border=self.ignore_border,
                st=self.stride,
                padding=self.padding
            )
