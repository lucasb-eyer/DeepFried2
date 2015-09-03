import DeepFried2 as df


class Log(df.Module):
    def symb_forward(self, symb_input):
        return df.T.log(symb_input)
