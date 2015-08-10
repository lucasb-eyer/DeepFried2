# -*- coding: utf-8 -*-
from .Optimizer import Optimizer
from ..utils import create_param_state_as

from theano.tensor import sqrt


class RMSProp(Optimizer):
    """
    Implements Hinton's "RMSProp" method presented in his Coursera lecture 6.5.
    Essentially, it sits right in-between AdaGrad and AdaDelta by being a
    windowed version of AdaGrad.

    The updates are:

        g²_{e+1} = ρ * g²_e + (1-ρ) * ∇p_e²
        p_{e+1} = p_e - (lr / √g²_{e+1}) * ∇p_e

    Note that in this case just initializing with epsilon is not enough anymore
    as we might get zero-gradient for some units long enough to completely fill
    the window.
    """

    def __init__(self, lr, rho, eps=1e-7):
        Optimizer.__init__(self, lr=lr, rho=rho, eps=eps)

    def get_updates(self, params, grads, lr, rho, eps):
        updates = []

        for param, grad in zip(params, grads):
            g2_state = create_param_state_as(param)
            new_g2 = rho*g2_state + (1-rho)*grad*grad
            updates.append((g2_state, new_g2))
            updates.append((param, param - lr/sqrt(new_g2+eps) * grad))

        return updates
