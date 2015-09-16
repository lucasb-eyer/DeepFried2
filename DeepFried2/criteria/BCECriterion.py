import DeepFried2 as df


class BCECriterion:
    """
    Like cross-entropy but also penalizing label-zero predictions.
    """

    def __init__(self, clip=None):
        self.clip = clip

    def symb_forward(self, symb_input, symb_targets):
        # A classic mistake, at least for myself.
        assert symb_targets.ndim == symb_input.ndim, "The targets of `{}` should have the same dimensionality as the net's output. You likely want to do something like `tgt[:,None]`.".format(df.typename(self))

        if self.clip is not None:
            symb_input = df.T.clip(symb_input, self.clip, 1-self.clip)
        return df.T.mean(df.T.sum(df.T.nnet.binary_crossentropy(symb_input, symb_targets), axis=1))
