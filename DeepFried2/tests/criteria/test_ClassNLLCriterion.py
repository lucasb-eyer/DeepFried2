#!/usr/bin/env python3

import DeepFried2 as df

import unittest
import numpy as np

class TestClassNLLCriterion(unittest.TestCase):

    def setUp(self):
        self.X = np.full((6,3), 0.1, dtype=df.floatX)
        self.T = np.array([0,1,2,2,1,0], dtype=np.int32)
        self.nll = -np.log(0.1)

    def testBasic(self):
        nll = df.ClassNLLCriterion().forward(self.X)
        np.testing.assert_almost_equal(nll, self.nll)

    def testAxis(self):
        X = self.X.reshape(3,2,3)
        T = self.T.reshape(3,2)
        nll = df.ClassNLLCriterion(classprob_axis=-1)
        nll = nll.forward(X, T)
        np.testing.assert_almost_equal(nll, self.nll)
