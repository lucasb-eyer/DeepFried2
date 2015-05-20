from .Layers import Module

class Container(Module):

    def __init__(self, *modules):
        super().__init__()

        self.modules = []
        for module in modules:
            self.add(module)

    def evaluate(self):
        super().evaluate()
        for module in self.modules:
            module.evaluate()

    def training(self):
        super().training()
        for module in self.modules:
            module.training()

    def parameters(self):
        params, grads = [], []

        for module in self.modules:
            mod_params, mod_grads = module.parameters()
            params += mod_params
            grads += mod_grads

        return params, grads

    def add(self, module):
        self.modules.append(module)

    def symbolic_forward(self, symbolic_input):
        raise NotImplementedError
