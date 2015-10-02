import DeepFried2 as df


def spatial_softmax(img, algo, mode):
    # check if we have 3D input
    if img.ndim == 5:
        """symb_input shape: (n_input, depth, channels, height, width)"""
        # shape: bdc01
        vol_shape = img.shape
        # shape: bcd01
        vol = df.T.basic.swapaxes(img, 1, 2)
        # shape: bcd(0*1)
        vol = df.T.flatten(vol, outdim=4)
        vol = df.th.sandbox.cuda.basic_ops.gpu_contiguous(vol)
        res = df.th.sandbox.cuda.dnn.GpuDnnSoftmax(tensor_format='bc01', algo=algo, mode=mode)(vol)
        # shape: bdc(0*1)
        res = df.T.basic.swapaxes(res, 1, 2)
        # shape: bdc01
        return res.reshape(vol_shape)
    else:
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
