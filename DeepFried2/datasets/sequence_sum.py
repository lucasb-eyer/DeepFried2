import numpy as _np
from DeepFried2 import floatX as _floatX

def data(T, N, nterms=2):
    """
    `T` is the sequence length, `N` is the number of sequences and `nterms` is
    the amount of terms that should be "active" in the stream.

    Returned array is of shape (N, T, 2)
    """
    X = _np.random.rand(N, T).astype(_floatX)

    mask = _np.zeros((N, T), _floatX)

    # Need to do this instead of just randint(0, T, (bs,nterms)) because we always need nterms distinct ones.
    for i in range(N):
        mask[i,_np.random.choice(T, size=nterms, replace=False)] = 1

    y = _np.sum(X[mask > 0].reshape((-1,nterms)), axis=1)
    X = _np.stack([X, mask], axis=-1)
    return X, y[:,None]
