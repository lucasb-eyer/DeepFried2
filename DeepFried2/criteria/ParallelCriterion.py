import DeepFried2 as df
from itertools import chain as _chain


class ParallelCriterion(df.Criterion):
    # TODO: Might actually want the weights to be shared variables so we can change their values on-the-fly!
    def __init__(self, *weighted_criteria, **kw):
        """
        Allowed keyword arguments are:

        - `repeat_target`:
            - `False` (default) means that as many targes need to be provided
              to `accumulate_gradients` as there are criteria.
            - `True` means that a single target needs to be provided, which
              will be used for all of the criteria.
        """
        df.Criterion.__init__(self)
        self.repeat_target = kw.get('repeat_target', False)
        self.criteria = []

        for wc in weighted_criteria:
            self.add(*df.utils.flatten(wc))

    def add(self, weight_or_crit, crit=None):
        if crit is None:
            self.criteria.append((1.0, weight_or_crit))
        else:
            self.criteria.append((weight_or_crit, crit))

    def symb_forward(self, symb_inputs, symb_targets):
        symb_inputs = list(symb_inputs)
        symb_targets = list(symb_targets)

        if self.repeat_target:
            symb_targets = symb_targets * len(symb_inputs)

        assert len(symb_inputs) == len(symb_targets) == len(self.criteria), "`{}` mismatch in number of inputs ({}), criteria ({}) and targets ({})" .format(df.utils.typename(self), len(symb_inputs), len(self.criteria), len(symb_targets))

        return sum(w*c(i, t) for (w,c), i, t in zip(self.criteria, symb_inputs, symb_targets))

    def get_extra_outputs(self):
        return list(_chain.from_iterable(c.get_extra_outputs() for (w,c) in self.criteria))
