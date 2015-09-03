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

    def get_stat_updates(self):
        stat_updates = []
        for module in self.modules:
            stat_updates += module.get_stat_updates()
        return stat_updates

    def add(self, *modules):
        assert all(isinstance(m, df.Module) for m in modules), "`{}`s can only contain objects subtyping `df.Module`.".format(self.__class__.__name__)
        self.modules += modules

    def __getitem__(self, slice_):
        return type(self)(*df.utils.aslist(self.modules[slice_]))
