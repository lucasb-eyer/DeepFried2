# -*- coding: utf-8 -*-
import DeepFried2 as df


class AdaGrad(df.Optimizer):
    """
    Implements Duchi's "Adaptive Subgradient" method, aka AdaGrad.
    Chris Dyer's "Notes on AdaGrad" are pretty awesome for practical purposes.

    TL;DR: AdaGrad doesn't need additional parameters (a lie) and makes the
           optimization much less sensitive to the learning-rate!

    In reality, it was a pioneer of fixing slow-learning features by adapting
    a feature's own learning-rate using an estimate of its raw 2nd moment, but
    its ideas have flown into superior AdaDelta and Adam.

    The updates are:

        g²_{e+1} = g²_e + ∇(p_e)²
        p_{e+1} = p_e - (lr / √g²_{e+1}) * ∇p_e

    that is, divide the learning-rate by a running square of the gradient.

    Note that this would lead to division by 0 in the beginning for those
    weights which don't receive a gradient (might be many with ReLUs), so we
    initialize g² with a small value.
    """

    def __init__(self, lr, eps=1e-7):
        df.Optimizer.__init__(self, lr=lr)

        # eps is only needed as numeric value for initializing state and it's
        # not possible to initialize state using symbolic variables.
        self.eps=eps

    def get_updates(self, params, grads, lr):
        updates = []

        for param, grad in zip(params, grads):
            g2_state = df.utils.create_param_state_as(param, initial_value=self.eps)
            new_g2 = g2_state + grad*grad
            updates.append((g2_state, new_g2))
            updates.append((param, param - lr/df.T.sqrt(new_g2) * grad))

        return updates
