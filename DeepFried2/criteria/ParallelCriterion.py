class ParallelCriterion:
    # TODO: Might actually want the weights to be shared variables so we can change their values on-the-fly!
    def __init__(self, *weighted_criteria, repeat_target=False, penalties=None):
        self.repeat_target = repeat_target
        self.criteria = []
        self.penalties = []

        for wc in weighted_criteria:
            self.add(*wc)
        for wp in penalties:
            self.add_penalty(*wp)

    def _add(self, cont, weight_or_crit, crit=None):
        if crit is None:
            weight, crit = 1.0, weight_or_crit
        else:
            weight, crit = weight_or_crit, crit

        cont.append((weight, crit))

    def add(self, weight_or_crit, crit=None):
        self._add(self.criteria, weight_or_crit, crit)

    def add_penalty(self, weight_or_crit, crit=None):
        self._add(self.penalties, weight_or_crit, crit)

    def symb_forward(self, symb_inputs, symb_targets):
        if self.repeat_target:
            symb_targets = [symb_targets] * len(symb_inputs)

        assert len(symb_inputs) == len(symb_targets) == len(self.criteria), "`{}` mismatch in number of inputs ({}), criteria ({}) and targets ({})" .format(df.typename(self), len(symb_inputs), len(self.criteria), len(symb_targets))

        return sum(w*c.symb_forward(i, t) for (w,c), i, t in zip(self.criteria, symb_inputs, symb_targets)) \
             + sum(w*p.symb_forward() for (w,p) in self.penalties)
