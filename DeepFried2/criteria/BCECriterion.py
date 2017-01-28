import DeepFried2 as df


class BCECriterion(df.Criterion):
    """
    Like cross-entropy but also penalizing label-zero predictions directly.
    """

    def __init__(self, clip=None, sumaxis=None):
        """
        - clip: clip inputs to [clip, 1-clip] to avoid potential numerical issues.
        - sumaxis: if we want to sum along one or more axes to get a per-sample
                   BCE in case each sample is made of more than one BCE
                   (e.g. each pixel in an image.)
        """
        df.Criterion.__init__(self)
        self.clip = clip
        self.sumaxis = sumaxis

    def symb_forward(self, symb_input, symb_target):
        self._assert_same_dim(symb_input, symb_target)

        if self.clip is not None:
            symb_input = df.T.clip(symb_input, self.clip, 1-self.clip)

        bce = df.T.nnet.binary_crossentropy(symb_input, symb_target)

        if self.sumaxis is not None:
            bce = df.T.sum(bce, self.sumaxis)

        return bce
