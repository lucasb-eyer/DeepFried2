from .Module import Module


class Identity(Module):

    def symb_forward(self, symb_input):
        return symb_input
