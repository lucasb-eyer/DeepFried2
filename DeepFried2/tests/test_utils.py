#!/usr/bin/env python3

import DeepFried2 as df2

import unittest
import numpy as np

class TestUtils(unittest.TestCase):

    def testFlatten(self):
        self.assertListEqual(df2.utils.flatten(1), [1])

        self.assertListEqual(df2.utils.flatten([1,2]), [1,2])
        self.assertListEqual(df2.utils.flatten((1,2)), [1,2])

        self.assertListEqual(df2.utils.flatten([[1,2],[None,4]]), [1,2,None,4])
        self.assertListEqual(df2.utils.flatten(((1,2),(None,4))), [1,2,None,4])
        self.assertListEqual(df2.utils.flatten([[1,[2,[None,[4,]]]]]), [1,2,None,4])
        self.assertListEqual(df2.utils.flatten([(1,(2,(None,(4,))))]), [1,2,None,4])
        self.assertListEqual(df2.utils.flatten([[[[1],2],None],4]), [1,2,None,4])
        self.assertListEqual(df2.utils.flatten([(([1],2),None),4]), [1,2,None,4])

        self.assertEqual(df2.utils.flatten(None, none_to_empty=True), [])
        self.assertEqual(df2.utils.flatten(None, none_to_empty=False), [None])

        self.assertListEqual(df2.utils.flatten([[1,2],[None,4]], none_to_empty=True), [1,2,4])
        self.assertListEqual(df2.utils.flatten([[1,[2,[None,[4,]]]]], none_to_empty=True), [1,2,4])
        self.assertListEqual(df2.utils.flatten([[[[1],2],None],4], none_to_empty=True), [1,2,4])

    def testFreezingThawing(self):
        l = df2.Linear(2,3)
        self.assertIn(l.W, l.parameters(learnable_only=True))

        df2.utils.freeze(l)
        self.assertEqual(0, len(l.parameters(learnable_only=True)))

        df2.utils.thaw(l)
        self.assertIn(l.W, l.parameters(learnable_only=True))
