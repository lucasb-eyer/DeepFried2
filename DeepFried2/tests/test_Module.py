#!/usr/bin/env python3

import DeepFried2 as df

import unittest
import numpy as np

class TestModule(unittest.TestCase):

    def testDifferentCriteriaInstances(self):
        T = np.random.randn(10,10).astype(df.floatX)
        c1 = df.MSECriterion()
        c2 = df.MADCriterion()
        err = 0.5

        net = df.Identity()
        l1 = float(net.accumulate_gradients(T+err, T, c1))
        l2 = float(net.accumulate_gradients(T+err, T, c2))

        np.testing.assert_almost_equal(l1, err**2)
        np.testing.assert_almost_equal(l2, abs(err))
