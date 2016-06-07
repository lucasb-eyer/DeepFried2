import DeepFried2 as df


class BCECriterion(df.Criterion):
    """
    Like cross-entropy but also penalizing label-zero predictions directly.
    """

    def __init__(self, clip=None):
        """
        - clip: clip inputs to [clip, 1-clip] to avoid potential numerical issues.
        """
        df.Criterion.__init__(self)
        self.clip = clip

    def symb_forward(self, symb_input, symb_target):
        self._assert_same_dim(symb_input, symb_target)

        if self.clip is not None:
            symb_input = df.T.clip(symb_input, self.clip, 1-self.clip)

        return df.T.nnet.binary_crossentropy(symb_input, symb_target)
