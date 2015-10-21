import DeepFried2 as df


class BCECriterion(df.Criterion):
    """
    Like cross-entropy but also penalizing label-zero predictions directly.
    """

    def __init__(self, clip=None):
        df.Criterion.__init__(self)
        self.clip = clip

    def symb_forward(self, symb_input, symb_targets):
        self._assert_same_dim(symb_input, symb_targets)

        if self.clip is not None:
            symb_input = df.T.clip(symb_input, self.clip, 1-self.clip)
        return df.T.mean(df.T.sum(df.T.nnet.binary_crossentropy(symb_input, symb_targets), axis=1))
