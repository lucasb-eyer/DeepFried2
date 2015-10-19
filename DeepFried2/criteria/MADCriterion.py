import DeepFried2 as df


class MADCriterion:
    def symb_forward(self, symb_input, symb_target):
        return df.T.mean(abs(symb_input - symb_target))
