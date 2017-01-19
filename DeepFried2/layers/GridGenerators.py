import DeepFried2 as df


class AttentionGrid(df.Module):
    def __init__(self, gh, gw, static=False):
        df.Module.__init__(self)
        self.dy = 2/(gh-1)
        self.dx = 2/(gw-1)

        if static:
            self.theta = self._addparam((1,4), self.bias_init(), name='θatt', decay=False,
                                        broadcastable=[True, False])
            self._theta = self.theta.param

    def symb_forward(self, theta):
        # In the static case, we use our param, not the input.
        theta = getattr(self, '_theta', theta)
        sx, sy, tx, ty = theta[:,0], theta[:,1], theta[:,2], theta[:,3]

        # At the time of writing, Theano only supports scalars here :(
        # I could(should) extend it to multi-dimensional some day.
        gy, gx = df.T.mgrid[-1:1.00001:self.dy, -1:1.00001:self.dx]
        return df.T.concatenate([
            df.T.cast(gx, df.floatX)[None,None,:,:]*sx[:,None,None,None]+tx[:,None,None,None],
            df.T.cast(gy, df.floatX)[None,None,:,:]*sy[:,None,None,None]+ty[:,None,None,None],
        ], axis=1)

    def bias_init(self):
        return df.init.const([1,1,0,0])


class AffineGrid(df.Module):
    def __init__(self, gh, gw, static=False):
        df.Module.__init__(self)
        self.dy = 2/(gh-1)
        self.dx = 2/(gw-1)

        if static:
            self.theta = self._addparam((1,2,3), self.bias_init(), name='θaff', decay=False,
                                        broadcastable=[True, False, False])
            self._theta = self.theta.param

    def symb_forward(self, theta):
        # In the static case, we use our param, not the input.
        # Otherwise, possibly reshape a flattened input, e.g. coming from a `Linear`.
        theta = getattr(self, '_theta', theta.reshape((-1, 2, 3)))

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        gx, gy, g1 = df.T.mgrid[-1:1.00001:self.dx, -1:1.00001:self.dy, 1:2]
        grid = df.T.cast(df.T.concatenate([gx,gy,g1], axis=2), df.floatX)

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        grid = df.T.tensordot(theta, grid, [theta.ndim-1,grid.ndim-1])

        # From (W,H) ordering back to (H,W) ordering.
        return grid.dimshuffle(0, 1, 3, 2)

    def bias_init(self):
        return df.init.const([1,0,0,
                              0,1,0])


class ProjectiveGrid(df.Module):
    def __init__(self, gh, gw, static=False):
        df.Module.__init__(self)
        self.dy = 2/(gh-1)
        self.dx = 2/(gw-1)

        if static:
            self.theta = self._addparam((1,8), self.bias_init(), name='θaff', decay=False)
            self._theta = df.T.concatenate(
                [self.theta.param, df.T.ones((1,1))], axis=-1
            ).reshape((1,3,3))

    def symb_forward(self, theta):
        # In the static case, we use our param, not the input.
        # Otherwise, possibly reshape a flattened input, e.g. coming from a `Linear`.
        if hasattr(self, '_theta'):
            theta = self._theta
        else:
            theta = theta.reshape((-1,8))
            theta = df.T.concatenate(
                [theta, df.T.ones((theta.shape[0], 1))], axis=-1
            ).reshape((-1,3,3))

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        gx, gy, g1 = df.T.mgrid[-1:1.00001:self.dx, -1:1.00001:self.dy, 1:2]
        grid = df.T.cast(df.T.concatenate([gx,gy,g1], axis=2), df.floatX)

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s, w_s)
        grid_homog = df.T.tensordot(theta, grid, [theta.ndim-1,grid.ndim-1])

        # From (W,H) ordering back to (H,W) ordering.
        grid_homog = grid_homog.dimshuffle(0, 1, 3, 2)

        # From homogenous coordinates back to plain ones.
        return grid_homog[:,:2,:,:] / df.T.addbroadcast(grid_homog[:,2:,:,:], 1)

    def bias_init(self):
        return df.init.const([1,0,0,
                              0,1,0,
                              0,0])
