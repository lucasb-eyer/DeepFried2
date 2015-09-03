import DeepFried2 as df


class Identity(df.Module):

    def symb_forward(self, symb_input):
        return symb_input
