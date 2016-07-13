#!/usr/bin/env python3

import DeepFried2 as df

import unittest
import numpy as np

class TestParam(unittest.TestCase):

    def testFreezingThawing(self):
        p = df.Param((2,2), df.init.const(0))

        self.assertTrue(p.learnable())
        p.freeze()
        self.assertFalse(p.learnable())
        p.thaw()
        self.assertTrue(p.learnable())

    def testFreezingDontLearn(self):
        l = df.Linear(2,3)
        W0 = l.W.get_value()

        X = np.random.randn(5,2).astype(df.floatX)
        Y = np.random.randn(5,3).astype(df.floatX)
        opt = df.SGD(0.1)

        l.W.freeze()
        l.zero_grad_parameters()
        l.accumulate_gradients(X, Y, df.MSECriterion())
        opt.update_parameters(l)
        np.testing.assert_array_equal(W0, l.W.get_value())

        l.clear()  # HACK, see github.com/lucasb-eyer/DeepFried2/issues/86
        opt.states = {}  # Same HACK again!!
        l.W.thaw()
        l.zero_grad_parameters()
        l.accumulate_gradients(X, Y, df.MSECriterion())
        opt.update_parameters(l)
        self.assertNotAlmostEqual(np.max(np.abs(W0 - l.W.get_value())), 0)
