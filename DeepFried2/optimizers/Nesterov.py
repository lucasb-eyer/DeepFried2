# -*- coding: utf-8 -*-
import DeepFried2 as df


class Nesterov(df.Optimizer):
    """
    Implementation of "Nesterov's Accelerated Gradient" (NAG) which is explained
    in further detail in

    "On the importance of initialization and momentum in deep learning"

    But the equation for NAG has been reshuffled by Nicolas Boulanger in

    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617

    for easier implementation in Theano. The updates are:

        v_{e+1} = mom * v_e - lr * ∇p_e
        p_{e+1} = p_e + mom * v_{e+1} - lr * ∇p_e
    """

    def __init__(self, lr, momentum):
        df.Optimizer.__init__(self, lr=lr, momentum=momentum)

    def get_updates(self, params, grads, lr, momentum):
        updates = []

        for param, grad in zip(params, grads):
            param_mom = df.utils.create_param_state_as(param)
            v = momentum * param_mom - lr * grad
            updates.append((param_mom, v))
            updates.append((param, param + momentum * v - lr * grad))

        return updates
