import DeepFried2 as df

from itertools import chain


class L1WeightDecay:
    def __init__(self, *containers):
        self.containers = containers

    def symb_forward(self):
        # TODO: Not sure if unique or non-unique make more sense!
        params = (c.unique_parameters()[0] for c in self.containers)
        return sum(df.T.sum(abs(p)) for p in chain(*params))


class L2WeightDecay:
    def __init__(self, *containers):
        self.containers = containers

    def symb_forward(self):
        # TODO: Not sure if unique or non-unique make more sense!
        params = (c.unique_parameters()[0] for c in self.containers)
        return sum(df.T.sum(p**2) for p in chain(*params))
