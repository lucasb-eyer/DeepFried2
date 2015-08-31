#!/usr/bin/env python3

import DeepFried2 as df2

import unittest
import numpy as np

class TestModule(unittest.TestCase):

    def testSISO(self):
        X = np.array([[1,2],[3,4]], dtype=df2.floatX)
        Y = df2.Identity().forward(X)
        np.testing.assert_array_equal(X, Y)

    def testMIMO(self):
        X = np.array([[1,2],[3,4]], dtype=df2.floatX)
        Y1, Y2 = df2.Identity().forward([X, X*2])
        np.testing.assert_array_equal(X, Y1)
        np.testing.assert_array_equal(X*2, Y2)
