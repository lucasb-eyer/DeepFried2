#!/usr/bin/env python3

import DeepFried2 as df

import unittest
import numpy as np

class TestSpatialMaxPooling3D(unittest.TestCase):

    def setUp(self):
        # Test the simplest case first, of 1 minibatch-entry and 1 data-channel.
        # (2,4,6)
        # From this, we can build up all other tests.
        self.X = np.array([
            # 1st depth slice
            [
                [ 1,  2,  3,  4, 57, 58, 59, 60],
                [ 5,  6,  7,  8, 53, 54, 55, 56],
                [ 9, 10, 11, 12, 49, 50, 51, 52],
                [48, 47, 46, 45, 88, 87, 86, 85],
                [44, 43, 42, 41, 92, 91, 90, 89],
                [40, 39, 38, 37, 96, 95, 94, 93],
            # 2nd depth slice
            ], [
                [13, 14, 15, 16, 69, 70, 71, 72],
                [17, 18, 19, 20, 65, 66, 67, 68],
                [21, 22, 23, 24, 61, 62, 63, 64],
                [36, 35, 34, 33, 76, 75, 74, 73],
                [32, 31, 30, 29, 80, 79, 78, 77],
                [28, 27, 26, 25, 84, 83, 82, 81],
            ]
        ], dtype=df.floatX)

        self.Z = np.array([
            [
                [24, 72],
                [48, 96]
            ]
        ], dtype=df.floatX)

    def testBasic(self):
        X = self.X[None,None,:,:,:]
        Z = self.Z[None,None,:,:,:]
        P = df.SpatialMaxPooling((2,3,4)).forward(X)
        np.testing.assert_array_equal(P, Z)

    def testMiniBatches(self):
        X = self.X[None,None,:,:,:]
        X = np.concatenate((X, X+1), axis=0)
        Z = self.Z[None,None,:,:,:]
        Z = np.concatenate((Z, Z+1), axis=0)

        P = df.SpatialMaxPooling((2,3,4)).forward(X)
        np.testing.assert_array_equal(P, Z)

    def testDataChannels(self):
        X = self.X[None,None,:,:,:]
        X = np.concatenate((X, X+1), axis=1)
        Z = self.Z[None,None,:,:,:]
        Z = np.concatenate((Z, Z+1), axis=1)

        P = df.SpatialMaxPooling((2,3,4)).forward(X)
        np.testing.assert_array_equal(P, Z)

    def testIgnoreBorder(self):
        X = np.pad(self.X, ((0,1),(0,2),(0,3)), mode='constant', constant_values=999)
        X = X[None,None,:,:,:]
        ZT = self.Z[None,None,:,:,:]
        ZF = np.pad(self.Z, ((0,1),(0,1),(0,1)), mode='constant', constant_values=999)
        ZF = ZF[None,None,:,:,:]
        P = df.SpatialMaxPooling((2,3,4), ignore_border=True).forward(X)
        np.testing.assert_array_equal(P, ZT)
        P = df.SpatialMaxPooling((2,3,4), ignore_border=False).forward(X)
        np.testing.assert_array_equal(P, ZF)

    def testStride(self):
        # Add another slice along depth
        X = np.concatenate((self.X, self.X[1:2]+100), axis=0)[None,None,:,:,:]
        Z = np.array([
            [
                [24, 70, 72],
                [48, 88, 88],
                [48, 92, 92],
                [48, 96, 96],
            ], [
                [124, 170, 172],
                [136, 176, 176],
                [136, 180, 180],
                [136, 184, 184],
            ],
        ], dtype=df.floatX)[None,None,:,:,:]
        P = df.SpatialMaxPooling((2,3,4), (1,1,2)).forward(X)
        np.testing.assert_array_equal(P, Z)
