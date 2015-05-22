import theano.sandbox.cuda.dnn as _dnn

from .Module import Module


class SpatialMaxPoolingCUDNN(Module):
    def __init__(self, k_w, k_h, d_w=None, d_h=None, pad_w=0, pad_h=0):
        Module.__init__(self)
        self.k_w = k_w
        self.k_h = k_h

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
        return _dnn.dnn_pool(
            img=symb_input,
            ws=(self.k_w, self.k_h),
            stride=(self.d_w, self.d_h),
            mode='max',
            pad=(self.pad_w, self.pad_h)
        )