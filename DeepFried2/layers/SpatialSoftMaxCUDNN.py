import DeepFried2 as df


def spatial_softmax(img, algo, mode):
    img = df.th.sandbox.cuda.basic_ops.gpu_contiguous(img)
    return df.th.sandbox.cuda.dnn.GpuDnnSoftmax(tensor_format='bc01', algo=algo, mode=mode)(img)


class SpatialSoftMaxCUDNN(df.Module):
    def __init__(self, algo='accurate', mode='channel'):
        # algo: 'fast' is straightforward softmax, 'accurate' is shifting inputs to avoid overflow.
        # mode: 'instance' is a softmax per image (across C,W,H), 'channel' is a softmax per pixel per image (across C).
        df.Module.__init__(self)
        self.algo = algo
        self.mode = mode

    def symb_forward(self, symb_input):
        return spatial_softmax(symb_input, self.algo, self.mode)
