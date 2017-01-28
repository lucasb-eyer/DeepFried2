import DeepFried2 as df


class SpatialTransformer(df.Module):
    def __init__(self, nearest=False):
        """
        Takes two inputs to forward: a (B,C,H,W) image `im` and a (B,2,H,W) `grid`
        of coordinates. Will sample `im` at each point of `grid` such as to
        apply an arbitrary transform described by `grid`.

        NOTE that when using `nearest`, backprop is not possible since the result
        is a piecewise constant function which thus has no gradient.

        Ref: "Spatial Transformer Networks" - Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
        """

        df.Module.__init__(self)
        self.nearest = nearest

    def symb_forward(self, symb_inputs):
        im, grids = symb_inputs

        B, C, H, W = im.shape
        h_f = df.T.cast(H, df.floatX)
        w_f = df.T.cast(W, df.floatX)

        # clip coordinates to [-1, 1]
        grids = df.T.clip(grids, -1, 1)

        # scale coordinates from [-1, 1] to [0, width/height - 1]
        # The weird [3:1:-1] means [shape[-1], shape[-2]], i.e. [W, H]
        # This is why the first grid component is w/x and second is h/y.
        whlim = (im.shape[3:1:-1] - 1)[None,:,None,None]
        grids = (grids + 1)/2 * whlim

        # Use this to index the batch-dimension, similar to typical NLL indexing.
        Ho, Wo = grids.shape[2:]
        Bidx = df.T.repeat(df.T.arange(B), Wo*Ho).reshape((-1,Ho,Wo))

        if self.nearest:
            # TODO: Remove this as a gradient through it cannot exist?
            #       But it can still be useful for forward-only!
            grids = df.T.cast(grids+0.5, 'uint32')  # Round to nearest index.
            newim = im[Bidx,:,grids[:,1,:,:],grids[:,0,:,:]]
        else:
            # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates,
            # grids0 being those top-left of it and grids1 those bottom-right.
            grids0 = df.T.cast(grids, 'uint32')       # Floor
            grids1 = df.T.minimum(grids0 + 1, whlim)  # Ceil

            x, x0, x1 = grids[:,0,:,:], grids0[:,0,:,:], grids1[:,0,:,:]
            y, y0, y1 = grids[:,1,:,:], grids0[:,1,:,:], grids1[:,1,:,:]
            i00 = im[Bidx,:,y0,x0]
            i10 = im[Bidx,:,y1,x0]
            i01 = im[Bidx,:,y0,x1]
            i11 = im[Bidx,:,y1,x1]

            # Note that we need to compute *all* weights from the lower side
            # or we might run into problems (all-zero weights) on the borders!
            x, x0 = df.T.cast(x, df.floatX), df.T.cast(x0, df.floatX)
            y, y0 = df.T.cast(y, df.floatX), df.T.cast(y0, df.floatX)
            w00 = ((1-(y-y0)) * (1-(x-x0)))[:,:,:,None]
            w01 = ((1-(y-y0)) *    (x-x0) )[:,:,:,None]
            w10 = (   (y-y0)  * (1-(x-x0)))[:,:,:,None]
            w11 = (   (y-y0)  *    (x-x0) )[:,:,:,None]
            newim = w00*i00 + w10*i10 + w01*i01 + w11*i11

        # The fancy indexing threw all "extra" dimensions (i.e. channels) to the end,
        # so here we need to move them back to where they belong.
        return newim.dimshuffle(0, 3, 1, 2)
