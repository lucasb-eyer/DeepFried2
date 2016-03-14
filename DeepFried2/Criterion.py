import DeepFried2 as df
from DeepFried2.utils import make_tensor_or_tensors, flatten


class Criterion(object):

    def __init__(self):
        self.penalties = []
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

    def full_symb_forward(self, symb_input, symb_target):
        cost = self.symb_forward(symb_input, symb_target)

        for w, p in self.penalties:
            cost += w*p.symb_forward()

        return cost

    def forward(self, num_input, num_target, with_penalties=True):
        # NOTE: using the GPU for such trivial computations as most costs
        # is actually somewhat slower (e.g. for RMSE: 1.2ms vs. 0.2ms). So
        # ideally, we'd like to compile a CPU-version here, but I don't know how!
        if with_penalties not in self._fn_forward:
            symb_in = make_tensor_or_tensors(num_input, 'Y')
            symb_tgt = make_tensor_or_tensors(num_target, 'T')
            if with_penalties:
                symb_out = self.full_symb_forward(symb_in, symb_tgt)
            else:
                symb_out = self.symb_forward(symb_in, symb_tgt)
            self._fn_forward[with_penalties] = df.th.function(
                inputs=flatten(symb_in) + flatten(symb_tgt),
                outputs=symb_out
            )

        return self._fn_forward[with_penalties](*(flatten(num_input) + flatten(num_target)))
