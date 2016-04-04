import DeepFried2 as df
import numpy as np


class RMSECriterion(df.Criterion):
    def __init__(self, axis='all', eps=1e-8):
        """
        - axis: 'all' means all except the batch-axis.
                Use `None` for literally all and `[]` for no summation.
        """
        df.Criterion.__init__(self)
        self.axis = axis
        self.eps = eps

    def symb_forward(self, symb_input, symb_target):
        self._assert_same_dim(symb_input, symb_target)

        axis = self.axis
        if self.axis == 'all':
            axis = np.arange(1, symb_input.ndim)

        return df.T.sqrt(self.eps + df.T.sum((symb_input - symb_target)**2, axis=axis))


class MSECriterion(df.Criterion):
    def symb_forward(self, symb_input, symb_target):
        self._assert_same_dim(symb_input, symb_target)
        return (symb_input - symb_target)**2
