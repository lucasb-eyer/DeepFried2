import DeepFried2 as df
from DeepFried2.utils import flatten

import numpy as _np


class BatchNormalization(df.Module):
    def __init__(self, n_features, eps=1e-5):
        """
        - `n_features` may be an integer (#features, #feature-maps for images) or a tuple.
            - If a single integer, it indicates the size of the 1-axis, i.e. first feature-axis.
              This is the only axis that will be normalized using statistics across all other axes.
            - If a tuple, it indicates the sizes of multiple axes (starting at 1) which are
              considered feature-axes and will consequently be normalized over statistics across all other axes.
        - `eps` is a small number which is added to the variance in order to
          avoid computing sqrt(0) for features with zero variance.
        """
        df.Module.__init__(self)

        self.ndim = len(flatten(n_features))

        self.W = self._addparam(n_features, df.init.const(1), name='W_BN_{}'.format(n_features))
        self.b = self._addparam(n_features, df.init.const(0), name='b_BN_{}'.format(n_features), decay=False)

        self.Winf = self._addparam(n_features, df.init.const(1), name='W_BN_{}_inf'.format(n_features), learn=False)
        self.binf = self._addparam(n_features, df.init.const(0), name='b_BN_{}_inf'.format(n_features), learn=False)

        # These are buffers for collecting the minibatch statistics.
        self.buf_var = df.th.shared(_np.full(n_features, 1, df.floatX), name='BN_var_{}'.format(n_features))
        self.buf_mean = df.th.shared(_np.full(n_features, 0, df.floatX), name='BN_mean_{}'.format(n_features))
        self.buf_count = df.th.shared(_np.asarray(0, dtype=df.floatX), name='BN_count_{}'.format(n_features))

        self.eps = eps or 1e-5

        self.batch_mean = None
        self.batch_var = None

    def symb_forward(self, symb_input):
        # Over which axis to normalize. This is at least 0 (batch-dimension)...
        axis = [0]

        # ...then, do not normalize over the `self.shape` dimensions but do over
        # the remaining ones. Take for example 2D images, for which we also
        # want to normalize over the spatial dimensions, e.g. (2,3).
        axis += list(range(self.ndim+1, symb_input.ndim))

        # And for the dimshuffle, similar story. Put 'x' on the axes we're normalizing.
        d_shuffle = ['x'] + list(range(self.ndim)) + ['x']*(symb_input.ndim-self.ndim-1)
        # Shorthand:
        def dshuf(x):
            return x.dimshuffle(*d_shuffle)

        # For example, for the usual case of images where dimensions are
        # (B,C,H,W), axis == [0, 2, 3] and d_shuffle == ['x', 0, 'x', 'x']

        if self.training_mode:
            self.batch_mean = df.T.mean(symb_input, axis=axis)
            self.batch_var = df.T.var(symb_input, axis=axis)

            symb_input = (symb_input - dshuf(self.batch_mean)) / dshuf(df.T.sqrt(self.batch_var + self.eps))

            return symb_input * dshuf(self.W.param) + dshuf(self.b.param)
        else:
            return symb_input * dshuf(self.Winf.param) + dshuf(self.binf.param)

    def get_stat_updates(self):
        assert (self.batch_mean is not None) and (self.batch_var is not None), "You need to do a forward pass first"

        # Update buffer statistics with current batch's statistics.
        return [
            (self.buf_mean, (self.buf_mean * self.buf_count + self.batch_mean) / (self.buf_count + 1.0)),
            (self.buf_var, (self.buf_var * self.buf_count + self.batch_var) / (self.buf_count + 1.0)),
            (self.buf_count, self.buf_count + 1.0),
        ]

    def training(self):
        df.Module.training(self)
        self.buf_count.set_value(0)
        self.batch_mean = None
        self.batch_var = None

    def evaluate(self):
        df.Module.evaluate(self)
        self.Winf.set_value(self.W.get_value() / _np.sqrt(self.buf_var.get_value() + self.eps))
        self.binf.set_value(self.b.get_value() - self.Winf.get_value() * self.buf_mean.get_value())

    def __getstate__(self):
        regular = df.Module.__getstate__(self)
        return [buf.get_value() for buf in (self.buf_mean, self.buf_var, self.buf_count)] + regular

    def __setstate__(self, state):
        istate = iter(state)
        for buf, val in zip((self.buf_mean, self.buf_var, self.buf_count), istate):
            buf.set_value(val)
        df.Module.__setstate__(self, istate)
