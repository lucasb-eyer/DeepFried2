import theano.tensor as _T


class RMSECriterion:
    def symb_forward(self, symb_input, symb_target):
        return _T.mean(_T.sqrt(_T.sum((symb_input - symb_target)**2, axis=1)))


class MSECriterion:
    def symb_forward(self, symb_input, symb_target):
        return _T.mean(_T.sum((symb_input - symb_target)**2, axis=1))
