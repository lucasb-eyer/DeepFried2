import DeepFried2 as df


class RepeatInput(df.Container):
    def symb_forward(self, symb_input):
        return tuple(module(symb_input) for module in self.modules)
