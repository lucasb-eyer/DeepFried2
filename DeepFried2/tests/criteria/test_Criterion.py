#!/usr/bin/env python3

import DeepFried2 as df

import unittest
import numpy as np

class TestCriterion(unittest.TestCase):

    def setUp(self):
        self.X = np.random.randn(10,10,10).astype(df.floatX)

    def testPerSampleForward(self):
        c = df.MSECriterion().enable_per_sample_cost()

        C = c.forward(self.X, self.X)
        np.testing.assert_equal(C, 0.0)

        C = c.forward(self.X, self.X, per_sample=True)
        np.testing.assert_equal(C.shape, self.X.shape)

    def testPerSampleAccum(self):
        c1 = df.MSECriterion().enable_per_sample_cost()
        net = df.Identity()
        net.accumulate_gradients(self.X, self.X, c1)
        C = c1.last_per_sample_cost()
        np.testing.assert_equal(C.shape, self.X.shape)

        c2 = df.MSECriterion()
        net = df.Identity()
        net.accumulate_gradients(self.X, self.X, c2)
        with self.assertRaises(ValueError):
            c2.last_per_sample_cost()

        c = df.ParallelCriterion(c1, c2, repeat_target=True)
        net = df.Parallel(df.Identity(), df.Identity())
        net.accumulate_gradients(self.X, self.X, c)
        C = c1.last_per_sample_cost()
        np.testing.assert_equal(C.shape, self.X.shape)
        with self.assertRaises(ValueError):
            c2.last_per_sample_cost()
