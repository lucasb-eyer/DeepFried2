import theano.tensor as _T


class ClassNLLCriterion:
    def symb_forward(self, symb_input, symb_targets):
        if symb_targets.ndim == 1:
            # This is the case when `symb_targets` are 1-hot encoding indices.
            int_targets = _T.cast(symb_targets, 'int32')
            return _T.mean(-_T.log(symb_input[_T.arange(symb_targets.shape[0]), int_targets]))
        elif symb_targets.ndim == symb_input.ndim:
            # This is the case when both are full distributions.
            return _T.mean(-_T.sum(symb_targets * _T.log(symb_input), axis=symb_input.ndim-1))
        else:
            assert False, "Mismatch in dimensionalities of `ClassNLLCriterion` input and targets."
