import theano as _th
import theano.tensor as _T


class Module:

    def __init__(self):
        self.training_mode = True

        # The functions are stored in a dictionary whose keys correspond to the
        # values that `self.training_mode` can take. That way, it would be
        # trivial to extend to further modes, and the code avoids many branches.
        self.fn_forward = {}
        self.fn_accum_grads = {}
        self.fn_accum_stats = {}

    #def __hash__(self):
    #    raise NotImplementedError("You *need* to reimplement hash, even if it's just python's default. See the documentation for more info.")

    def zero_grad_parameters(self):
        _, grads = self.parameters()
        for grad in grads:
            grad.set_value(0 * grad.get_value())

    def parameters(self):
        params, grads = [], []

        if hasattr(self, 'weight'):
            assert hasattr(self, 'grad_weight'), "The layer {} has a `weight` variable but no `grad_weight`, you probably forget to implement it.".format(type(self))
            params += [self.weight]
            grads += [self.grad_weight]

        if hasattr(self, 'bias'):
            assert hasattr(self, 'grad_bias'), "The layer {} has a `bias` variable but no `grad_bias`, you probably forget to implement it.".format(type(self))
            params += [self.bias]
            grads += [self.grad_bias]

        return params, grads

    def evaluate(self):
        self.training_mode = False

    def training(self):
        self.training_mode = True

    def symb_forward(self, symb_input):
        raise NotImplementedError

    def forward(self, data):
        if self.training_mode not in self.fn_forward:
            symb_in = _T.TensorType(_th.config.floatX, (False,) * data.ndim)('X')
            symb_out = self.symb_forward(symb_in)
            self.fn_forward[self.training_mode] = _th.function(
                inputs=[symb_in],
                outputs=symb_out
            )

        return self.fn_forward[self.training_mode](data)

    def accumulate_gradients(self, data_in, data_tgt, loss):
        if self.training_mode not in self.fn_accum_grads:
            symb_in = _T.TensorType(_th.config.floatX, (False,) * data_in.ndim)('X')
            symb_tgt = _T.TensorType(_th.config.floatX, (False,) * data_tgt.ndim)('T')
            symb_out = self.symb_forward(symb_in)
            symb_err = loss.symb_forward(symb_out, symb_tgt)

            params, grads = self.parameters()
            symb_grads = _th.grad(cost=symb_err, wrt=params)

            grads_updates = [(grad, grad + symb_grad) for grad, symb_grad in zip(grads, symb_grads)]
            self.fn_accum_grads[self.training_mode] = _th.function(
                inputs=[symb_in, symb_tgt],
                outputs=symb_err,
                updates=grads_updates
            )

        return self.fn_accum_grads[self.training_mode](data_in, data_tgt)

    def get_stat_updates(self):
        return []

    def accumulate_statistics(self, data_in):
        if self.training_mode not in self.fn_accum_stats:
            symb_in = _T.TensorType(_th.config.floatX, (False,) * data_in.ndim)('X')
            self.symb_forward(symb_in)

            stat_updates = self.get_stat_updates()
            if not stat_updates:
                # If there's no layer collecting statistics, we don't need to
                # compile and call a function. This prevents theano errors.
                return

            self.fn_accum_stats[self.training_mode] = _th.function(
                inputs=[symb_in],
                updates=stat_updates
            )

        self.fn_accum_stats[self.training_mode](data_in)
