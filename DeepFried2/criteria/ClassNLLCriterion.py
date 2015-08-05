import theano.tensor as _T


class ClassNLLCriterion:
    def symb_forward(self, symb_input, symb_targets):
        int_targets = _T.cast(symb_targets, 'int32')
        return _T.mean(-_T.log(symb_input[_T.arange(symb_targets.shape[0]), int_targets]))
