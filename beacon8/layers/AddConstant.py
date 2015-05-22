from .Module import Module


class AddConstant(Module):
    def __init__(self, scalar):
        self.scalar = scalar

    def symb_forward(self, symb_input):
        return symb_input + self.scalar
