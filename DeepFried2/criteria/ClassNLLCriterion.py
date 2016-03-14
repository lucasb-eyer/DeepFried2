import DeepFried2 as df
from functools import reduce as _reduce


def _prod(l):
    return _reduce(lambda x,y: x*y, l, 1)


class ClassNLLCriterion(df.Criterion):
    """
    Contrary to Torch7, this criterion takes raw probabilities as input and
    relies on Theano's graph-optimizations to generate numerically stable code.

    This might need to change to require log-probabilities when making use of
    cuDNN's softmax in the near future.

    ClassNLLCriterion has two modus operandi, which might be split into two
    criteria in the future:

    1. If the input has the same number of dimensions as the targets:
        - The targets are considered to be probabilities.
        - Computes element-wise cross-entropy, taking the log of the input.
    2. If the input has one more dimension than the targets:
        - The targets are considered to be one-hot encoded class labels.

    This condition might also take the target dtype into account in the future.
    """
    def __init__(self, clip=None, classprob_axis=1):
        """
        - `clip`: if not `None`, clips the incoming probabilites into the range
            [`clip`, 1-`clip`] in order to avoid numerical instabilities of the
            `log` operation. This is not necessary in the 1-hot case.

        - `classprob_axis`: The axis along which the class-probabilities reside,
            i.e. this axis should have the same length as number of classes.
        """
        df.Criterion.__init__(self)
        self.clip = clip
        self.axis = classprob_axis

    def symb_forward(self, symb_input, symb_targets):
        D = symb_input.ndim

        if symb_targets.ndim == D - 1:
            # 1-hot encoding case.

            int_targets = df.T.cast(symb_targets, 'int32')  # No-op if already int32.
            ax = (D + self.axis) % D  # Compute meaning of negative axis index.

            # Build the indexing operation that corresponds to the N-dimensional
            # generalization of the classical y[arange(N), T]
            # Also build up the shape of the result at the same time.
            idx = []
            sha = []
            for i in range(D):
                if i == ax:
                    idx.append(int_targets.flatten())
                else:
                    nbefore = _prod(symb_input.shape[j] for j in range(i) if j != ax)
                    nafter  = _prod(symb_input.shape[j] for j in range(i+1, D) if j != ax)
                    idx.append(df.T.tile(df.T.arange(symb_input.shape[i]).repeat(nafter), nbefore))
                    sha.append(symb_input.shape[i])

            # Extract only those entries indicated by the 1-hot encoding
            # and then reshape to the original shape, dropping `self.axis`.
            p_y = symb_input[tuple(idx)].reshape(sha)

            if self.clip is not None:
                p_y = df.T.clip(p_y, self.clip, 1-self.clip)

            return -df.T.log(p_y)

        elif symb_targets.ndim == D:
            # This is the case when both are full distributions.

            p_y = symb_input
            if self.clip is not None:
                p_y = df.T.clip(p_y, self.clip, 1-self.clip)
            return -df.T.sum(symb_targets * df.T.log(p_y), axis=self.axis)

        else:
            assert False, "Mismatch in dimensionalities of `{}` input ({}) and targets ({}).".format(df.utils.typename(self), symb_input.ndim, symb_targets.ndim)
