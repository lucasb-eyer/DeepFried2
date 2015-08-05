from .Module import Module


class ReLU(Module):

    def symb_forward(self, symb_input):
        return (symb_input + abs(symb_input)) * 0.5
