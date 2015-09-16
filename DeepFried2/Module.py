import DeepFried2 as df
from DeepFried2.utils import make_tensor_or_tensors, aslist

from collections import OrderedDict as _OrderedDict
import numpy as _np

class Module:

    def __init__(self):
        self.training_mode = True

        # The functions are stored in a dictionary whose keys correspond to the
        # values that `self.training_mode` can take. That way, it would be
        # trivial to extend to further modes, and the code avoids many branches.
        self._fn_forward = {}
        self._fn_accum_grads = {}
        self._fn_accum_stats = {}

    #def __hash__(self):
    #    raise NotImplementedError("You *need* to reimplement hash, even if it's just python's default. See the documentation for more info.")

    def zero_grad_parameters(self):
        _, grads = self.unique_parameters()  # Here, it's just a matter of performance. But even then, not really.
        for grad in grads:
            grad.set_value(_np.zeros_like(grad.get_value()))

    def parameters(self):
        params, grads = [], []

        if hasattr(self, 'weight'):
            assert hasattr(self, 'grad_weight'), "The layer {} has a `weight` variable but no `grad_weight`, you probably forget to implement it.".format(df.classname(self))
            params += [self.weight]
            grads += [self.grad_weight]

        if hasattr(self, 'bias'):
            assert hasattr(self, 'grad_bias'), "The layer {} has a `bias` variable but no `grad_bias`, you probably forget to implement it.".format(df.classname(self))
            params += [self.bias]
            grads += [self.grad_bias]

        return params, grads

    def unique_parameters(self):
        # We actually need to remove duplicates from the list of parameters
        # (and their corresponding gradients) in order to support reusing
        # the same layer at multiple places in the graph,
        # e.g. do weight sharing.
        params, grads = self.parameters()
        return (
            list(_OrderedDict.fromkeys(params).keys()),
            list(_OrderedDict.fromkeys(grads).keys()),
        )

    def may_decay(self):
        flags = []
        if hasattr(self, 'weight'):
            flags += [True]
        if hasattr(self, 'bias'):
            flags += [False]
        return flags

    def evaluate(self):
        self.training_mode = False

    def training(self):
        self.training_mode = True

    def symb_forward(self, symb_input):
        raise NotImplementedError("`{}` needs to implement `symb_forward` method.".format(df.typename(self)))

    def forward(self, data):
        if self.training_mode not in self._fn_forward:
            symb_in = make_tensor_or_tensors(data, 'X')
            symb_out = self.symb_forward(symb_in)
            self._fn_forward[self.training_mode] = df.th.function(
                inputs=aslist(symb_in),
                outputs=symb_out
            )

        return self._fn_forward[self.training_mode](*aslist(data))

    def accumulate_gradients(self, data_in, data_tgt, loss):
        if self.training_mode not in self._fn_accum_grads:
            symb_in = make_tensor_or_tensors(data_in, 'X')
            symb_tgt = make_tensor_or_tensors(data_tgt, 'T')
            symb_out = self.symb_forward(symb_in)
            symb_err = loss.full_symb_forward(symb_out, symb_tgt)

            params, grads = self.unique_parameters()
            symb_grads = df.th.grad(cost=symb_err, wrt=params)

            grads_updates = [(grad, grad + symb_grad) for grad, symb_grad in zip(grads, symb_grads)]
            self._fn_accum_grads[self.training_mode] = df.th.function(
                inputs=aslist(symb_in) + aslist(symb_tgt),
                outputs=symb_err,
                updates=grads_updates
            )

        args = aslist(data_in) + aslist(data_tgt)
        return self._fn_accum_grads[self.training_mode](*args)

    def get_stat_updates(self):
        return []

    def accumulate_statistics(self, data_in):
        if self.training_mode not in self._fn_accum_stats:
            symb_in = make_tensor_or_tensors(data_in, 'X')

            # Call forward once so it can compute some variables it'll actually
            # use in the stat updates collection.
            self.symb_forward(symb_in)

            stat_updates = self.get_stat_updates()
            if not stat_updates:
                # If there's no layer collecting statistics, we don't need to
                # compile and call a function. This prevents theano errors.
                return

            # Need to make sure there's only one update per variable for the
            # case where we've got the same module instance at multiple places
            # within the graph.
            # Also warn about it because it's not obvious whether just dropping
            # one of them is the right thing to do in general?
            todo = set(upd[0] for upd in stat_updates)
            if len(todo) < len(stat_updates):
                uniq_updates = []
                for upd in stat_updates:
                    if upd[0] in todo:
                        uniq_updates.append(upd)
                        todo.remove(upd[0])
                    else:
                        print("WARNING: Dropped the following stat-update because that variable got multiple updates: {}".format(upd[0]))
                stat_updates = uniq_updates

            self._fn_accum_stats[self.training_mode] = df.th.function(
                inputs=aslist(symb_in),
                updates=stat_updates
            )

        self._fn_accum_stats[self.training_mode](*aslist(data_in))

    def clear(self):
        self._fn_forward.clear()
        self._fn_accum_grads.clear()
        self._fn_accum_stats.clear()
