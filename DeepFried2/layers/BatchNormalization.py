import DeepFried2 as df
from DeepFried2.utils import create_param, create_param_and_grad

import numpy as _np


class BatchNormalization(df.Module):
    def __init__(self, n_features, eps=None):
        df.Module.__init__(self)

        self.weight, self.grad_weight = create_param_and_grad(n_features, df.init.const(1), name='W_BN_{}'.format(n_features))
        self.bias, self.grad_bias = create_param_and_grad(n_features, df.init.const(0), name='b_BN_{}'.format(n_features))

        self.inference_weight = create_param(n_features, df.init.const(1), name='W_BN_{}_inf'.format(n_features))
        self.inference_bias = create_param(n_features, df.init.const(0), name='b_BN_{}_inf'.format(n_features))

        # These are buffers for collecting the minibatch statistics.
        self.buffer_variance = create_param(n_features, df.init.const(1), name='BN_var_{}'.format(n_features))
        self.buffer_mean = create_param(n_features, df.init.const(0), name='BN_mean_{}'.format(n_features))
        self.buffer_counts = df.th.shared(_np.asarray(0, dtype=df.floatX), name='BN_count_{}'.format(n_features))

        self.eps = eps or 1e-5

        self.batch_mean = None
        self.batch_var = None

    def symb_forward(self, symb_input):
        d_shuffle = ('x', 0)
        axis = (0,)

        if symb_input.ndim == 4:
            d_shuffle += ('x', 'x')
            axis += (2, 3)
        elif symb_input.ndim == 5:
            d_shuffle = ('x', 'x', 0, 'x', 'x')
            axis = (0, 1, 3, 4)

        if self.training_mode:
            self.batch_mean = df.T.mean(symb_input, axis=axis)
            self.batch_var = df.T.var(symb_input, axis=axis)

            return (symb_input - self.batch_mean.dimshuffle(*d_shuffle)) / df.T.sqrt(self.batch_var + self.eps).dimshuffle(*d_shuffle) * self.weight.dimshuffle(*d_shuffle) + self.bias.dimshuffle(*d_shuffle)
        else:
            return symb_input * self.inference_weight.dimshuffle(*d_shuffle) + self.inference_bias.dimshuffle(*d_shuffle)

    def get_stat_updates(self):
        assert (self.batch_mean is not None) and (self.batch_var is not None), "You need to do a forward pass first"

        stat_updates = list()
        stat_updates.append((self.buffer_mean,
                             (self.buffer_mean * self.buffer_counts + self.batch_mean) / (self.buffer_counts + 1.0)))

        stat_updates.append((self.buffer_variance,
                             (self.buffer_variance * self.buffer_counts + self.batch_var) / (self.buffer_counts + 1.0)))

        stat_updates.append((self.buffer_counts,
                             self.buffer_counts + 1.0))

        return stat_updates

    def training(self):
        df.Module.training(self)
        self.buffer_counts.set_value(0)
        self.batch_mean = None
        self.batch_var = None

    def evaluate(self):
        df.Module.evaluate(self)
        self.inference_weight.set_value(self.weight.get_value() / _np.sqrt(self.buffer_variance.get_value() + self.eps))
        self.inference_bias.set_value(self.bias.get_value() - self.inference_weight.get_value() * self.buffer_mean.get_value())
