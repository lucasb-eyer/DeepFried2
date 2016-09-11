import DeepFried2 as df


class Embedding(df.Module):
    def __init__(self, ntok, ndim, init=df.init.ortho_svd()):
        """A layer that learns `ntok` embedding vectors of dimension `ndim`.

        Note that this doesn't take care of any `unk`/`oov` token.
        If you need that, increase `ntok` by one and handle it from the outside.

        The input to this layer is an array of indices of any arbitrary shape.
        The output will be an array of the same shape plus an added dimension
        for the embeddings at the end (as last dimension).
        """
        df.Module.__init__(self)

        self.ndim = ndim
        self.W = self._addparam((ntok, ndim), init, name='Wemb_{}x{}'.format(ntok, ndim))

    def symb_forward(self, symb_input):
        return self.W.param[symb_input.flatten()].reshape(tuple(symb_input.shape) + (self.ndim,))
