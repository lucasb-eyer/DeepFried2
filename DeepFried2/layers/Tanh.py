import DeepFried2 as df


class Tanh(df.Module):

    def symb_forward(self, symb_input):
        return df.T.tanh(symb_input)
