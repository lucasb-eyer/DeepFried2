import DeepFried2 as df


class SoftMax(df.Module):
    def symb_forward(self, symb_input):
        return df.T.nnet.softmax(symb_input)
