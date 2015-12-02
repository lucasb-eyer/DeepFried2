import DeepFried2 as df


class Container(df.Module):

    def __init__(self, *modules):
        df.Module.__init__(self)

        self.modules = []
        self.add(*modules)

    def evaluate(self):
        df.Module.evaluate(self)
        for module in self.modules:
            module.evaluate()

    def training(self):
        df.Module.training(self)
        for module in self.modules:
            module.training()

    def parameters(self):
        params, grads = [], []

        for module in self.modules:
            mod_params, mod_grads = module.parameters()
            params += mod_params
            grads += mod_grads

        return params, grads

    def may_decay(self):
        return sum((m.may_decay() for m in self.modules), [])

    def get_stat_updates(self):
        stat_updates = []
        for module in self.modules:
            stat_updates += module.get_stat_updates()
        return stat_updates

    def add(self, *modules):
        for m in modules:
            assert isinstance(m, df.Module), "`{}`s can only contain objects subtyping `df.Module`. You tried to add the following `{}`: {}".format(df.typename(self), df.typename(m), m)
        self.modules += modules

    def __getitem__(self, slice_):
        return type(self)(*df.utils.aslist(self.modules[slice_]))

    def __getstate__(self):
        return [m.__getstate__() for m in self.modules]

    def __setstate__(self, state):
        for m, s in zip(self.modules, state):
            m.__setstate__(s)
