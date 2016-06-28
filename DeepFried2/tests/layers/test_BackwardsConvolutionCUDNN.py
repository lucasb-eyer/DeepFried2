#!/usr/bin/env python3

import DeepFried2 as df

import unittest
import numpy as np

class TestBackwardsConvolutionCUDNN(unittest.TestCase):

    def testFwdBwd(self):
        # Let's try fuzz-testing for that one, I hear good things about it!!

        randint = np.random.randint

        for _ in range(100):
            B = randint(1, 10)
            cin = randint(1, 10)
            cout = randint(1, 10)

            # Test both 2D and 3D
            ndim = randint(2,3)

            fs = tuple(randint(1, 11, size=ndim))

            # Image should be >= filter size in all dimensions.
            ims = tuple(fs + randint(0, 10, size=ndim))

            stride = tuple(randint(1, 4, size=ndim))
            border = tuple(randint(0, 5, size=ndim))

            # We can only test those cases where no border gets "lost" during
            # forward conv using this strategy, as otherwise the result is smaller.
            # I could come up with the formula of the output shape in other cases,
            # but why bother if brute-force trying is just as good?
            # A little over a third of trials pass this test.
            if not all(((i+2*b) - f) % s == 0 for f,i,s,b in zip(fs, ims, stride, border)):
                continue

            X = np.random.randn(B, cin, *ims).astype(df.floatX)
            net = df.Sequential(
                df.SpatialConvolutionCUDNN(cin, cout, fs, stride, border),
                df.BackwardsConvolutionCUDNN(cout, cin, fs, stride, border)
            )
            Y = net.forward(X)
            self.assertEqual(Y.shape, X.shape, "Setup: B={B},cin={cin},cout={cout},ndim={ndim},fs={fs},ims={ims},stride={stride},border={border}".format(**locals()))
