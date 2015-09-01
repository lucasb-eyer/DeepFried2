import theano.tensor as _T


class ClassNLLCriterion:
    def __init__(self, clip=None):
        self.clip = clip

    def symb_forward(self, symb_input, symb_targets):
        if symb_targets.ndim == 1:
            # This is the case when `symb_targets` are 1-hot encoding indices.
            int_targets = _T.cast(symb_targets, 'int32')
            p_y = symb_input[_T.arange(symb_targets.shape[0]), int_targets]
            if self.clip is not None:
                p_y = _T.clip(p_y, self.clip, 1-self.clip)
            return _T.mean(-_T.log(p_y))

        elif symb_targets.ndim == symb_input.ndim:
            # This is the case when both are full distributions.
            p_y = symb_input
            if self.clip is not None:
                p_y = _T.clip(p_y, self.clip, 1-self.clip)
            return _T.mean(-_T.sum(symb_targets * _T.log(p_y), axis=symb_input.ndim-1))

        else:
            assert False, "Mismatch in dimensionalities of `ClassNLLCriterion` input and targets."
