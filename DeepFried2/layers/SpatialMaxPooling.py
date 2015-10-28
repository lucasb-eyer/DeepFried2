import DeepFried2 as df
from theano.tensor.signal.downsample import max_pool_2d


class SpatialMaxPooling(df.Module):
    def __init__(self, k_w, k_h, d_w=None, d_h=None, pad_w=0, pad_h=0, ignore_border=False):
        df.Module.__init__(self)
        self.k_w = k_w
        self.k_h = k_h
        self.ignore_border = ignore_border

        if d_w is None:
            self.d_w = self.k_w
        else:
            self.d_w = d_w

        if d_h is None:
            self.d_h = self.k_h
        else:
            self.d_h = d_h

        self.pad_w = pad_w
        self.pad_h = pad_h

    def symb_forward(self, symb_input):
        return max_pool_2d(
            symb_input,
            ds=(self.k_h, self.k_w),
            ignore_border=self.ignore_border,
            st=(self.d_h, self.d_w),
            padding=(self.pad_h, self.pad_w)
        )
