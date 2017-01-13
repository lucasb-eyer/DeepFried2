import DeepFried2 as df


class Add(df.Module):
    def symb_forward(self, symb_inputs):
        assert isinstance(symb_inputs, (list, tuple)), "Input to `Add` must be multiple tensors."

        s = symb_inputs[0]
        for x in symb_inputs[1:]:
            s = s + x
        return s


def block(nchan, fs=(3,3), body=None):
    return df.Sequential(
        df.RepeatInput(
            df.Sequential(
                df.BatchNormalization(nchan), df.ReLU(),
                df.SpatialConvolutionCUDNN(nchan, nchan, fs, border='same', init=df.init.prelu(), bias=False),
                df.BatchNormalization(nchan), df.ReLU(),
                df.SpatialConvolutionCUDNN(nchan, nchan, fs, border='same', init=df.init.prelu(), bias=False)
            ) if body is None else body,
            df.Identity()
        ),
        Add()
    )


def block_proj(nin, nout, fs=(3,3), body=None):
    return df.Sequential(
        df.RepeatInput(
            df.Sequential(
                df.BatchNormalization(nin), df.ReLU(),
                df.SpatialConvolutionCUDNN(nin, nout, fs, border='same', init=df.init.prelu(), bias=False),
                df.BatchNormalization(nout), df.ReLU(),
                df.SpatialConvolutionCUDNN(nout, nout, fs, border='same', init=df.init.prelu(), bias=False)
            ) if body is None else body,
            df.SpatialConvolutionCUDNN(nin, nout, (1,)*len(fs)),
        ),
        Add()
    )
