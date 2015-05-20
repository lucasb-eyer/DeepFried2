from .Container import Container


class Sequential(Container):
    def symb_forward(self, symb_input):
        symb_output = symb_input
        for module in self.modules:
            symb_output = module.symb_forward(symb_output)
        return symb_output