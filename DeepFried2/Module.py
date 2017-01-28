import DeepFried2 as df
from DeepFried2.utils import tensors_for_ndarrays, flatten

import numpy as _np

class Module(object):

    def __init__(self):
        self._mode = 'train'

        # The functions are stored in a dictionary whose keys correspond to the
        # values that `self._mode` can take.
        self._fn_forward = {}
        self._fn_accum_grads = {}
        self._fn_accum_stats = {}

        # These will store the last gotten/produced symbolic input/output
        # expressions, respectively. The key is the current mode.
        self._last_symb_inp = {}
        self._last_symb_out = {}

    #def __hash__(self):
    #    raise NotImplementedError("You *need* to reimplement hash, even if it's just python's default. See the documentation for more info.")

    def _addparam(self, shape, init, *a, **kw):
        assert init is not None and init is not False, "`{}` requires parameter `{}` to have initializer.".format(df.utils.typename(self), kw.get("name", "unnamed"))

        # Add it here because many don't even have params. This avoids misuse.
        if not hasattr(self, '_params'):
            self._params = []

        param = df.Param(shape, init, *a, **kw)
        self._params.append(param)
        return param

    def _addparam_optional(self, shape, init, *a, **kw):
        if init is None or init is False:
            return None

        return self._addparam(shape, init, *a, **kw)


    def zero_grad_parameters(self):
        for p in self.parameters(learnable_only=True):
            p.zero_grad()

    def parameters(self, learnable_only=False):
        params = getattr(self, '_params', [])
        if learnable_only:
            params = [p for p in params if p.learnable()]
        return params

    def evaluate(self):
        self._mode = 'eval'

    def training(self):
        self._mode = 'train'

    def symb_forward(self, symb_input):
        raise NotImplementedError("`{}` needs to implement `symb_forward` method.".format(df.utils.typename(self)))

    def __call__(self, symb_input):
        # Keep track of the symbolic inputs/outputs for things such as `Backward` layer.
        self._last_symb_inp[self._mode] = symb_input
        self._last_symb_out[self._mode] = self.symb_forward(symb_input)
        return self._last_symb_out[self._mode]

    def forward(self, data):
        if self._mode not in self._fn_forward:
            symb_in = tensors_for_ndarrays(data, 'X')
            symb_out = self(symb_in)
            extra_out = self.get_extra_outputs()
            extra_up = self.get_extra_updates()
            fn = self._fn_forward[self._mode] = df.th.function(
                inputs=flatten(symb_in),
                outputs=flatten(symb_out) + flatten(extra_out),
                updates=flatten(extra_up, types=list),
            )
            fn._df2_extra = extra_out

        fn = self._fn_forward[self._mode]
        outs = fn(*flatten(data))
        return self._collect_extra_outputs(fn, outs)

    def accumulate_gradients(self, data_in, data_tgt, crit):
        if (self._mode, id(crit)) not in self._fn_accum_grads:
            symb_in = tensors_for_ndarrays(data_in, 'X')
            symb_tgt = tensors_for_ndarrays(data_tgt, 'T')
            symb_out = self(symb_in)
            symb_cost = crit(symb_out, symb_tgt)
            extra_out = self.get_extra_outputs() + crit.get_extra_outputs()
            extra_up = self.get_extra_updates()

            params = self.parameters(learnable_only=True)
            symb_grads = df.th.grad(cost=symb_cost, wrt=[p.param for p in params])
            grads_updates = [(p.grad, p.grad + symb_grad) for p, symb_grad in zip(params, symb_grads)]

            fn = self._fn_accum_grads[self._mode, id(crit)] = df.th.function(
                inputs=flatten(symb_in) + flatten(symb_tgt),
                outputs=flatten(symb_cost) + flatten(extra_out),
                updates=grads_updates + flatten(extra_up, types=list),
            )
            fn._df2_extra = extra_out

        fn = self._fn_accum_grads[self._mode, id(crit)]
        args = flatten(data_in) + flatten(data_tgt)
        outs = fn(*args)
        return self._collect_extra_outputs(fn, outs)

    def get_extra_outputs(self):
        """
        Return a list of Theano expressions which will be passed as additional
        `output` parameters. The computed value will be stored in the
        expression's `val` attribute.

        Guaranteed to be called after `symb_forward`.
        """
        return []

    def get_extra_updates(self):
        """NOTE: MUST BE LIST OF TUPLES (because of how flatten is called)"""
        return []

    def _collect_extra_outputs(self, fn, vals):
        # The number of non-extra outputs.
        nout = len(vals) - len(fn._df2_extra)

        # Store all outputs in the `val` attribute so that they can possibly
        # be retrieved by the modules that asked for them.
        for out, val in zip(fn._df2_extra, vals[nout:]):
            out.val = val

        return vals[:nout] if nout > 1 else vals[0]

    def get_stat_updates(self):
        """
        Return extra `update` statements, currently only Batch-Normalization.
        As soon as something else has a similar need, we might need to unify
        them, and possibly change/generalize the training/evaluate modes.

        Guaranteed to be called after `symb_forward`.
        """
        return []

    def accumulate_statistics(self, data_in):
        if self._mode not in self._fn_accum_stats:
            symb_in = tensors_for_ndarrays(data_in, 'X')

            # Call forward once so it can compute some variables it'll actually
            # use in the stat updates collection.
            self(symb_in)

            stat_updates = self.get_stat_updates()
            if not stat_updates:
                # If there's no layer collecting statistics, we don't need to
                # compile and call a function. This prevents theano errors.
                return
            extra_up = self.get_extra_updates()

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

            self._fn_accum_stats[self._mode] = df.th.function(
                inputs=flatten(symb_in),
                updates=stat_updates + flatten(extra_up, types=list)
            )

        self._fn_accum_stats[self._mode](*flatten(data_in))

    def clear(self):
        self._fn_forward.clear()
        self._fn_accum_grads.clear()
        self._fn_accum_stats.clear()

    def __getstate__(self):
        return [p.get_value() for p in self.parameters()]

    def __setstate__(self, state):
        params = self.parameters()
        if len(params) != len(state):
            raise ValueError("{} wants to load {} params but received {} params".format(df.utils.typename(self), len(params), len(state)))

        for p, s in zip(params, state):
            if p.get_value().shape != s.shape:
                raise ValueError("{} got invalid shape when loading param {}: expecting {} but loading {}".format(df.utils.typename(self), p.param.name, p.get_value().shape, s.shape))

            p.set_value(s)
