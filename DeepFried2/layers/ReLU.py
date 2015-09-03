import DeepFried2 as df


class ReLU(df.Module):

    def symb_forward(self, symb_input):
        return (symb_input + abs(symb_input)) * 0.5
