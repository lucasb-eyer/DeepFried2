from .Container import Container

import theano.tensor as _T


class Parallel(Container):
    def symb_forward(self, symb_input):
        # TODO: Not sure if this polymorphism is any good!
        if isinstance(symb_input, (list, tuple)):
            assert len(symb_input) == len(self.modules), "If `Parallel` has multiple inputs, it should be the same amount as it has modules."
            return tuple(module.symb_forward(symb_in) for module, symb_in in zip(self.modules, symb_input))
        else:
            return tuple(module.symb_forward(symb_input) for module in self.modules)
