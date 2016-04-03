import DeepFried2 as df
from DeepFried2.utils import tensors_for_ndarrays, flatten


class Criterion(object):

    def __init__(self):
        self.penalties = []
        self.with_weights = False
        self._fn_forward = {}

    def _assert_same_dim(self, symb_input, symb_target):
        # A classic mistake, at least for myself.
        assert symb_target.ndim == symb_input.ndim, "The targets of `{}` should have the same dimensionality as the net's output. You likely want to do something like `tgt[:,None]`.".format(df.utils.typename(self))

    def symb_forward(self, symb_input, symb_target):
        raise NotImplementedError("`{}` needs to implement `symb_forward` method.".format(df.utils.typename(self)))

    # TODO: Might actually want the weights to be shared variables so we can change their values on-the-fly!
    def add_penalty(self, weight_or_pen, pen=None):
        if pen is None:
            weight, pen = 1.0, weight_or_pen
        else:
            weight, pen = weight_or_pen, pen
        self.penalties.append((weight, pen))

    def full_symb_forward(self, symb_input, symb_target, with_penalties=True):
        # Possibly extract the weights as 2nd target.
        if self.with_weights is True:
            symb_target, symb_weights = symb_target
        # Or extract a 0/1 weighting using magic value.
        elif self.with_weights is not False:
            symb_weights = df.T.neq(symb_target, self.with_weights)
        else:
            symb_weights = None

        # TODO: Actually, don't
        cost = self.symb_forward(symb_input, symb_target)

        # TODO: Here, we can keep/store/output unweighted per-sample costs!

        if symb_weights is not None:
            cost = symb_weights * cost

        # TODO: Here, we can keep/store/output weighted per-sample costs!

        # Criteria may return per-sample cost which we will average
        # (optionally weighted) across samples.
        if cost.ndim != 0:
            cost = df.T.mean(cost)
            if symb_weights is not None:
                # Need a very small eps to avoid 0/0 when all weights are 0!
                cost = cost / (1e-8 + df.T.mean(symb_weights))

        if with_penalties:
            for w, p in self.penalties:
                cost += w*p.symb_forward()

        return cost

    def enable_weights(self):
        self.with_weights = True
        return self

    def enable_maskval(self, val):
        self.with_weights = val
        return self

    def forward(self, num_input, num_target, with_penalties=True):
        # NOTE: using the GPU for such trivial computations as most costs
        # is actually somewhat slower (e.g. for RMSE: GPU 1.2ms vs. CPU 0.2ms).
        # So ideally, we'd like to compile a CPU-version here, but I don't know how!
        if with_penalties not in self._fn_forward:
            symb_in = tensors_for_ndarrays(num_input, 'Y')
            symb_tgt = tensors_for_ndarrays(num_target, 'T')
            symb_out = self.full_symb_forward(symb_in, symb_tgt, with_penalties)
            self._fn_forward[with_penalties] = df.th.function(
                inputs=flatten(symb_in) + flatten(symb_tgt),
                outputs=symb_out
            )

        return self._fn_forward[with_penalties](*(flatten(num_input) + flatten(num_target)))
