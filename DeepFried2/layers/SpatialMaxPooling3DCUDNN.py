import DeepFried2 as df


class SpatialMaxPooling3DCUDNN(df.Module):
    def __init__(self, k_w, k_h, k_d, d_w=None, d_h=None, d_d=None, pad_w=0, pad_h=0, pad_d=0):
        df.Module.__init__(self)
        self.k_w = k_w
        self.k_h = k_h
        self.k_d = k_d

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

        self.pad_w = pad_w
        self.pad_h = pad_h
        self.pad_d = pad_d

    def symb_forward(self, symb_input):
        """symb_input shape: (n_input, channels, depth, height, width)"""

        if symb_input.ndim < 5:
            raise NotImplementedError('3D max pooling requires a dimension >= 5')

        return df.th.sandbox.cuda.dnn.dnn_pool(
            img=symb_input,
            ws=(self.k_d, self.k_h, self.k_w),
            stride=(self.d_d, self.d_h, self.d_w),
            mode='max',
            pad=(self.pad_d, self.pad_h, self.pad_d)
        )
