# -*- coding: utf-8 -*-
import DeepFried2 as df


class SGD(df.Optimizer):

    def __init__(self, lr):
        df.Optimizer.__init__(self, lr=lr)

    def get_updates(self, params, grads, lr):
        return [(p, p - lr * g) for p, g in zip(params, grads)]
