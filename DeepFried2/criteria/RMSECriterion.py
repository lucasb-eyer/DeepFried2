import DeepFried2 as df
import numpy as np


class RMSECriterion(df.Criterion):
    def symb_forward(self, symb_input, symb_target):
        self._assert_same_dim(symb_input, symb_target)

        return df.T.mean(df.T.sqrt(df.T.sum((symb_input - symb_target)**2, axis=np.arange(1,symb_input.ndim))))


class MSECriterion(df.Criterion):
    def symb_forward(self, symb_input, symb_target):
        self._assert_same_dim(symb_input, symb_target)

        return df.T.mean(df.T.sum((symb_input - symb_target)**2, axis=np.arange(1,symb_input.ndim)))
