import DeepFried2 as df


class MADCriterion(df.Criterion):
    def symb_forward(self, symb_input, symb_target):
        self._assert_same_dim(symb_input, symb_target)

        return abs(symb_input - symb_target)
