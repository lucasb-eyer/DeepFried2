from ..layers import Module


class Container(Module):

    def __init__(self, *modules):
        Module.__init__(self)

        self.modules = []
        self.add(*modules)

    def evaluate(self):
        Module.evaluate(self)
        for module in self.modules:
            module.evaluate()

    def training(self):
        Module.training(self)
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
        assert all(isinstance(m, Module) for m in modules), "`Container`s can only contain objects subtyping `Module`."
        self.modules += modules
