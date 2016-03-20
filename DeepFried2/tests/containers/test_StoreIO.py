#!/usr/bin/env python3

import DeepFried2 as df

import unittest
import numpy as np

class TestStoreIO(unittest.TestCase):

    def test(self):
        net = df.StoreIO(df.Linear(2,3))

        net.training()
        X = np.array([[1,2],[3,4]], dtype=df.floatX)
        Y = net.forward(X)
        np.testing.assert_array_equal(net.inp, X)
        np.testing.assert_array_equal(net.out, Y)

        net.evaluate()
        X = np.array([[10,20],[30,40]], dtype=df.floatX)
        Y = net.forward(X)
        np.testing.assert_array_equal(net.inp, X)
        np.testing.assert_array_equal(net.out, Y)

    def testMulti(self):
        net = df.StoreIO(df.Identity())

        net.training()
        X = np.array([[1,2],[3,4]], dtype=df.floatX)
        Y1, Y2 = net.forward([X, X*2])
        np.testing.assert_array_equal(net.inp[0], X)
        np.testing.assert_array_equal(net.inp[1], X*2)
        np.testing.assert_array_equal(net.out[0], Y1)
        np.testing.assert_array_equal(net.out[1], Y2)

        net.evaluate()
        X = np.array([[10,20],[30,40]], dtype=df.floatX)
        Y1, Y2 = net.forward([X, X*2])
        np.testing.assert_array_equal(net.inp[0], X)
        np.testing.assert_array_equal(net.inp[1], X*2)
        np.testing.assert_array_equal(net.out[0], Y1)
        np.testing.assert_array_equal(net.out[1], Y2)
