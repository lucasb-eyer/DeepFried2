#!/usr/bin/env python3

import DeepFried2 as df

import unittest
import numpy as np

class TestSpatialOverfeatRoll(unittest.TestCase):

    def testNice(self):

        # First with just one single slice.
        X = np.random.randn(1,1,8,6).astype(df.floatX)
        Y = np.array(df.SpatialOverfeatRoll().forward(X))

        self.assertEqual(Y.shape, (4,1,4,3))
        np.testing.assert_array_equal(Y[0], X[0,:, ::2, ::2])
        np.testing.assert_array_equal(Y[1], X[0,:, ::2,1::2])
        np.testing.assert_array_equal(Y[2], X[0,:,1::2, ::2])
        np.testing.assert_array_equal(Y[3], X[0,:,1::2,1::2])

        # Then, with multiple images with multiple slices
        X = np.random.randn(2,3,8,6).astype(df.floatX)
        Y = np.array(df.SpatialOverfeatRoll().forward(X))

        self.assertEqual(Y.shape, (8,3,4,3))
        for i in range(X.shape[0]):
            np.testing.assert_array_equal(Y[4*i+0], X[i,:, ::2, ::2])
            np.testing.assert_array_equal(Y[4*i+1], X[i,:, ::2,1::2])
            np.testing.assert_array_equal(Y[4*i+2], X[i,:,1::2, ::2])
            np.testing.assert_array_equal(Y[4*i+3], X[i,:,1::2,1::2])


    def testNondiv(self):

        # And now we're being evil: non-divisible sizes.
        X = np.random.randn(2,3,9,7).astype(df.floatX)
        Y = np.array(df.SpatialOverfeatRoll().forward(X))

        # In this case, the rolled result is zero-padded.

        self.assertEqual(Y.shape, (8,3,4+1,3+1))
        for i in range(X.shape[0]):
            np.testing.assert_array_equal(Y[4*i+0], X[i,:, ::2, ::2])
            np.testing.assert_array_equal(Y[4*i+1,:,:,:3], X[i,:, ::2,1::2])
            np.testing.assert_equal      (Y[4*i+1,:,:,3:], 0)
            np.testing.assert_array_equal(Y[4*i+2,:,:4,:], X[i,:,1::2, ::2])
            np.testing.assert_equal      (Y[4*i+2,:,4:,:], 0)
            np.testing.assert_array_equal(Y[4*i+3,:,:4,:3], X[i,:,1::2,1::2])
            np.testing.assert_equal      (Y[4*i+3,:,4:,:], 0)
            np.testing.assert_equal      (Y[4*i+3,:,:,3:], 0)


    def testUnroll(self):
        X = np.random.randn(2,3,8,6).astype(df.floatX)
        Y = np.array(df.SpatialOverfeatRoll().forward(X))
        Z = np.array(df.SpatialOverfeatUnroll().forward(Y))
        np.testing.assert_equal(X, Z)

        # NOTE: This cannot work with non-divisible sizes.
