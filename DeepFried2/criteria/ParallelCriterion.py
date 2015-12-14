import DeepFried2 as df


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
            self.add(*df.utils.aslist(wc))

    def add(self, weight_or_crit, crit=None):
        if crit is None:
            self.criteria.append((1.0, weight_or_crit))
        else:
            self.criteria.append((weight_or_crit, crit))

    def symb_forward(self, symb_inputs, symb_targets):
        symb_inputs = df.utils.aslist(symb_inputs)
        symb_targets = df.utils.aslist(symb_targets)

        if self.repeat_target:
            symb_targets = [symb_targets] * len(symb_inputs)

        assert len(symb_inputs) == len(symb_targets) == len(self.criteria), "`{}` mismatch in number of inputs ({}), criteria ({}) and targets ({})" .format(df.typename(self), len(symb_inputs), len(self.criteria), len(symb_targets))

        return sum(w*c.symb_forward(i, t) for (w,c), i, t in zip(self.criteria, symb_inputs, symb_targets))
