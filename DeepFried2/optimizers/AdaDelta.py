# -*- coding: utf-8 -*-
from .Optimizer import Optimizer
from ..utils import create_param_state_as

from theano.tensor import sqrt


class AdaDelta(Optimizer):
    """
    Implements Matt Zeiler's "Adaptive Learningrate" method, aka. AdaDelta.
    The paper itself is really neat, and both very convincing and practical.

    TL;DR: 1. AdaGrad quickly anneals, AdaDelta doesn't. (No proof.)
           2. AdaGrad *is* sensitive to learning-rate, AdaDelta not so much. (Table 1.)
           3. AdaDelta includes 2nd-order approximation. (3.2)

    The updates are:

        g²_{e+1} = ρ * g²_e + (1-ρ) * ∇p_e²
        up_{e+1} = √(d²_e / g²_{e+1}) * ∇p_e
        d²_{e+1} = ρ * d²_e + (1-ρ) * up²
        p_{e+1} = p_e - up_{e+1}

    As in RMSProp, we need to add epsilons in order to create stability.

    It turns out that the effective learning-rate will converge to 1 as the
    gradients decrease (and thus learning grinds to a halt). This could be used
    to check for convergence by a specialized trainer.

    The only reason `lr` is still there is this tweet by Alec Radford:

    https://twitter.com/AlecRad/status/543518744799358977

        @kastnerkyle @ogrisel @johnmyleswhite @tcovert Adadelta raw is finicky,
        shrinking its updates by 0.5 "just works" in my experience as well.
    """

    def __init__(self, rho, eps=1e-7, lr=1):
        Optimizer.__init__(self, rho=rho, eps=eps, lr=lr)

    def get_updates(self, params, grads, rho, eps, lr):
        updates = []

        for param, grad in zip(params, grads):
            g2_state = create_param_state_as(param, prefix='g2_')
            d2_state = create_param_state_as(param, prefix='d2_')

            new_g2 = rho*g2_state + (1-rho)*grad*grad
            up = lr * sqrt((d2_state+eps) / (new_g2+eps)) * grad
            new_d2 = rho*d2_state + (1-rho)*up*up

            updates.append((g2_state, new_g2))
            updates.append((param, param - up))
            updates.append((d2_state, new_d2))

        return updates
