import DeepFried2 as df

# NOTE: We intentionally don't make these inherit from df.Criterion as we don't
#       really want them to be used as standalone criteria.


class L1WeightDecay:
    def __init__(self, *modules):
        self.modules = modules

    def symb_forward(self):
        return sum(df.T.sum(abs(p)) for p in _collect_decayable_params(self.modules))


class L2WeightDecay:
    def __init__(self, *modules):
        self.modules = modules

    def symb_forward(self):
        return sum(df.T.sum(p**2) for p in _collect_decayable_params(self.modules))


def _collect_decayable_params(modules):
    return [p.param for c in modules for p in c.parameters() if p.may_decay()]
