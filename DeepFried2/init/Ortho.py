import numpy as _np

def ortho_qr(gain=_np.sqrt(2)):
    # gain should be set based on the activation function:
    # linear activations     g = 1 (or greater)
    # tanh activations       g > 1
    # ReLU activations       g = sqrt(2) (or greater)

    def init(shape, fan):
        # Note that this is not strictly correct.
        #
        # What we'd really want is for an initialization which reuses ortho
        # matrices across layers, but we can't have that with the current arch:
        #
        # From A. Saxe's comment in https://plus.google.com/+SoumithChintala/posts/RZfdrRQWL6u
        # > This initialization uses orthogonal matrices, but there’s a bit of
        # > subtlety when it comes to undercomplete layers—basically you need to
        # > make sure that the paths from the input layer to output layer, through
        # > the bottleneck, are preserved. This is accomplished by reusing parts of
        # > the same orthogonal matrices across different layers of the network.
        flat = (shape[0], _np.prod(shape[1:]))
        q1, _ = _np.linalg.qr(_np.random.randn(flat[0], flat[0]))
        q2, _ = _np.linalg.qr(_np.random.randn(flat[1], flat[1]))
        w = _np.dot(q1[:,:min(flat)], q2[:min(flat),:])
        return gain * w.reshape(shape)
    return init

def ortho_svd(gain=_np.sqrt(2)):
    # gain should be set based on the activation function:
    # linear activations     g = 1 (or greater)
    # tanh activations       g > 1
    # ReLU activations       g = sqrt(2) (or greater)

    def init(shape, fan):
        flat = (shape[0], _np.prod(shape[1:]))
        u, _, v = _np.linalg.svd(_np.random.randn(*flat), full_matrices=False)
        w = u if u.shape == flat else v
        return gain * w.reshape(shape)
    return init
