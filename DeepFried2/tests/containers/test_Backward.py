#!/usr/bin/env python3

import DeepFried2 as df

import unittest
import numpy as np

class TestBackward(unittest.TestCase):

    def testSimple(self):
        l = df.Linear(2,3, bias=False)
        W = l.parameters()[0].get_value()
        net = df.Sequential(l, df.Backward(l))

        # Make sure they share the weight matrix!
        self.assertEqual(len(net.parameters()), 1)
        self.assertEqual(net[0].parameters(), net[1].parameters())

        # Check whether it actually performs the backward operation.
        X = np.random.randn(4,2).astype(df.floatX)
        Y = net.forward(X)
        np.testing.assert_array_equal(Y, np.dot(np.dot(X, W), W.T))

        # And check that it also works in a different mode.
        net.evaluate()
        Y = net.forward(X)
        np.testing.assert_array_equal(Y, np.dot(np.dot(X, W), W.T))

    def testWrt(self):
        l1 = df.Linear(2,3, bias=False)
        l2 = df.Linear(2,3, bias=False)
        W1 = l1.parameters()[0].get_value()
        W2 = l2.parameters()[0].get_value()
        net = df.Sequential(l1, df.Backward(l2, wrt=l1))

        # Just really make sure they don't share parameters, like, fo' real!
        self.assertEqual(len(net.parameters()), 2)
        p1, p2 = net.parameters()
        self.assertNotEqual(p1, p2)
        self.assertFalse(np.all(p1.get_value() == p2.get_value()))

        # Check whether it performs the right (backward) op.
        X = np.random.randn(4,2).astype(df.floatX)
        Y = net.forward(X)
        np.testing.assert_array_equal(Y, np.dot(np.dot(X, W1), W2.T))

        # And again, in another mode too.
        net.evaluate()
        Y = net.forward(X)
        np.testing.assert_array_equal(Y, np.dot(np.dot(X, W1), W2.T))

    def testGraph(self):
        # Test whether it also works in a somewhat more "complex" net
        # which includes a few other layers. Only testing size here
        # since other stuff was tested before.
        l1 = df.Linear(3,4)
        b1 = df.Backward(l1)
        net = df.Sequential(df.Linear(3,3), l1, df.Linear(4,4), b1, df.Linear(3,3))

        X = np.random.randn(2,3).astype(df.floatX)
        self.assertEqual(net.forward(X).shape, (2,3))
