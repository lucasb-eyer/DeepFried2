from .Module import Module


class Reshape(Module):
    def __init__(self, *new_shape):
        Module.__init__(self)
        self.new_shape = new_shape

    def symb_forward(self, symb_input):
        return symb_input.reshape(self.new_shape)
