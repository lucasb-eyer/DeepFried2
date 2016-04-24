import DeepFried2 as df


def spatial_softmax(img, algo, mode):
    # check if we have 3D input
    if img.ndim == 5:
        # shape: bcd01
        vol_shape = img.shape
        # shape: bcd(0*1)
        vol = df.T.flatten(img, outdim=4)
        vol = df.th.sandbox.cuda.basic_ops.gpu_contiguous(vol)
        res = df.th.sandbox.cuda.dnn.GpuDnnSoftmax(tensor_format='bc01', algo=algo, mode=mode)(vol)
        # shape: bcd01
        return res.reshape(vol_shape)
    else:
        img = df.th.sandbox.cuda.basic_ops.gpu_contiguous(img)
        return df.th.sandbox.cuda.dnn.GpuDnnSoftmax(tensor_format='bc01', algo=algo, mode=mode)(img)


class SpatialSoftMaxCUDNN(df.Module):
    def __init__(self, algo='accurate', mode='channel'):
        """
        - `algo`: One of the following values:
            - `'fast'`: Straightforward softmax, beware numerical problems.
            - `'accurate'`: Use the "subtract max input" stabilization to avoid overflows.
            - `'log'`: Compute the log-softmax instead.
        - `mode`: One of the following values:
            - `'instance'`: Computes a single whole softmax per image (across C,H,W; output is B).
            - `'channel'`: Computs a softmax per-pixel per-image (across C; output is B,H,W).
        """
        df.Module.__init__(self)
        self.algo = algo
        self.mode = mode

    def symb_forward(self, symb_input):
        return spatial_softmax(symb_input, self.algo, self.mode)
