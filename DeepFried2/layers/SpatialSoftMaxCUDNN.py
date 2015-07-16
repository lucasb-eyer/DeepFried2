from .Module import Module

import theano.sandbox.cuda.dnn as _dnn
import theano.sandbox.cuda.basic_ops as _cuops


def spatial_softmax(img, algo, mode):
    img = _cuops.gpu_contiguous(img)
    return _dnn.GpuDnnSoftmax(tensor_format='bc01', algo=algo, mode=mode)(img)


class SpatialSoftMaxCUDNN(Module):
    def __init__(self, algo='accurate', mode='channel'):
        # algo: 'fast' is straightforward softmax, 'accurate' is shifting inputs to avoid overflow.
        # mode: 'instance' is a softmax per image (across C,W,H), 'channel' is a softmax per pixel per image (across C).
        Module.__init__(self)
        self.algo = algo
        self.mode = mode

    def symb_forward(self, symb_input):
        return spatial_softmax(symb_input, self.algo, self.mode)
