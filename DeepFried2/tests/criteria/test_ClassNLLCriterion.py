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
        nll = df.ClassNLLCriterion().forward(self.X, self.T)
        np.testing.assert_approx_equal(nll, self.nll, significant=5)

    def testMask(self):
        nll = df.ClassNLLCriterion().enable_maskval(-1)
        nll = nll.forward(self.X, np.array([-1,1,2,2,1,-1], dtype=np.int32))
        np.testing.assert_approx_equal(nll, self.nll, significant=5)

    def testWeights(self):
        nll = df.ClassNLLCriterion().enable_weights()
        nll = nll.forward(self.X, [np.array([ 2,1,2,2,1,2 ], dtype=np.uint32),
                                   np.array([0.,1,1,1,1,0.], dtype=df.floatX)])
        np.testing.assert_approx_equal(nll, self.nll, significant=5)

    def testAxis(self):
        X = self.X.reshape(3,2,3)
        T = self.T.reshape(3,2)
        nll = df.ClassNLLCriterion(classprob_axis=-1)
        nll = nll.forward(X, T)
        np.testing.assert_approx_equal(nll, self.nll, significant=5)

    def testAxisMask(self):
        X = self.X.reshape(3,2,3)
        nll = df.ClassNLLCriterion(classprob_axis=-1).enable_maskval(-1)
        nll = nll.forward(X, np.array([[-1,1],[2,2],[1,-1]], dtype=np.int32))
        np.testing.assert_approx_equal(nll, self.nll, significant=5)

    def testAxisWeights(self):
        X = self.X.reshape(3,2,3)
        nll = df.ClassNLLCriterion(classprob_axis=-1).enable_weights()
        nll = nll.forward(X, [np.array([[ 2,1],[2,2],[1,2 ]], dtype=np.uint32),
                              np.array([[0.,1],[1,1],[1,0.]], dtype=df.floatX)])
        np.testing.assert_approx_equal(nll, self.nll, significant=5)
