import DeepFried2 as df


class UpSample(df.Module):
    def __init__(self, upsample=(2,2), output_shape=None):
        """upsample determines how many upsampling steps are taken. (height,width) for 2D or (depth,height,width) for 3D
           output_shape can be used to crop the upsampled result to a desired size. (height,width) for 2D or (depth,height,width) for 3D
        """
        df.Module.__init__(self)
        self.upsample = upsample
        self.output_shape = output_shape

    def symb_forward(self, symb_input):
        """symb_input shape: 2D: (n_input, channels, height, width)
                             3D: (n_input, channels, depth, height, width)
        """

        if symb_input.ndim == 4 and len(self.upsample) != 2:
            raise NotImplementedError('A 4D input tensor requires 2D upsampling')
        elif symb_input.ndim == 5 and len(self.upsample) != 3:
            raise NotImplementedError('A 5D input tensor requires 3D upsampling')

        res = symb_input.repeat(self.upsample[0], axis=2)
        res = res.repeat(self.upsample[1], axis=3)

        if symb_input.ndim == 4:
            if self.output_shape is not None:
                res = res[:,:,:self.output_shape[0],:self.output_shape[1]]
        elif symb_input.ndim == 5:
            res = res.repeat(self.upsample[2], axis=4)
            if self.output_shape is not None:
                res = res[:,:,:self.output_shape[0],:self.output_shape[1],:self.output_shape[2]]

        return res
