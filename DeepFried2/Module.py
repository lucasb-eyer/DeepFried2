import DeepFried2 as df
from DeepFried2.utils import make_tensor_or_tensors, aslist

import numpy as _np

class Module(object):

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

    def _addparam(self, *a, **kw):
        # Add it here because many don't even have params. This avoids misuse.
        if not hasattr(self, '_params'):
            self._params = []

        param = df.Param(*a, **kw)
        self._params.append(param)
        return param

    def zero_grad_parameters(self):
        for p in self.parameters(trainable_only=True):
            p.zero_grad()

    def parameters(self, trainable_only=False):
        params = getattr(self, '_params', [])
        if trainable_only:
            params = [p for p in params if p.trainable()]
        return params

    def evaluate(self):
        self.training_mode = False

    def training(self):
        self.training_mode = True

    def symb_forward(self, symb_input):
        raise NotImplementedError("`{}` needs to implement `symb_forward` method.".format(df.utils.typename(self)))

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

            params = self.parameters(trainable_only=True)
            symb_grads = df.th.grad(cost=symb_err, wrt=[p.param for p in params])
            grads_updates = [(p.grad, p.grad + symb_grad) for p, symb_grad in zip(params, symb_grads)]

            self._fn_accum_grads[self.training_mode] = df.th.function(
                inputs=aslist(symb_in) + aslist(symb_tgt),
                outputs=symb_err,
                updates=grads_updates
            )

        args = aslist(data_in) + aslist(data_tgt)
        return self._fn_accum_grads[self.training_mode](*args)

    def get_stat_updates(self):
        """
        Return extra `update` statements, currently only Batch-Normalization.
        As soon as something else has a similar need, we might need to unify
        them, and possibly change/generalize the training/evaluate modes.

        Guaranteed to be called after `symb_forward`.
        """
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

    def __getstate__(self):
        return [p.get_value() for p in self.parameters()]

    def __setstate__(self, state):
        for p, s in zip(self.parameters(), state):
            p.set_value(s)
