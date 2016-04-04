import DeepFried2 as df


class Sequential(df.Container):
    def symb_forward(self, symb_input):
        symb_output = symb_input
        for module in self.modules:
            symb_output = module(symb_output)
        return symb_output
