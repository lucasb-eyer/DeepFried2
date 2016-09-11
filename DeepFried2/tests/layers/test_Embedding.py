#!/usr/bin/env python3

import DeepFried2 as df

import unittest
import numpy as np

class TestEmbedding(unittest.TestCase):

    def testForward(self):
        X = np.array([
            [0, 1, 2],
            [2, 1, 0],
        ])

        Z = np.array([
            [[1,0,0,0], [0,1,0,0], [0,0,1,0]],
            [[0,0,1,0], [0,1,0,0], [1,0,0,0]],
        ])

        Y = df.Embedding(ntok=3, ndim=4, init=df.init.eye()).forward(X)
        np.testing.assert_array_equal(Y, Z)
