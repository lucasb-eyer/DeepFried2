import DeepFried2 as df

# NOTE: We intentionally don't make these inherit from df.Criterion as we don't
#       really want them to be used as standalone criteria.


class L1WeightDecay:
    def __init__(self, *containers):
        self.containers = containers

    def symb_forward(self):
        return sum(df.T.sum(abs(p)) for p in collect_decayable_params(*self.containers))


class L2WeightDecay:
    def __init__(self, *containers):
        self.containers = containers

    def symb_forward(self):
        return sum(df.T.sum(p**2) for p in collect_decayable_params(*self.containers))


def collect_decayable_params(*containers):
    decay_params = []
    for c in containers:
        params, _ = c.unique_parameters()  # TODO: unique or non-unique?
        may = c.may_decay()

        assert len(params) == len(may), "Possible implementation bug in `{}.may_decay()`: {} parameters, but {} decay infos.".format(df.utils.typename(c), len(params), len(may))

        decay_params += [p for p,m in zip(params, may) if may]
    return decay_params
