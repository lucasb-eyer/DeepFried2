# -*- coding: utf-8 -*-
import DeepFried2 as df


class DQNProp(df.Optimizer):
    """
    RMSProp as described here on page 23:
    http://arxiv.org/pdf/1308.0850v5.pdf
    
    Also used by DeepMind here:
    https://sites.google.com/a/deepmind.com/dqn/
    In NeuralQLearner.lua

    The updates are:

        g_{e+1} = ρ * g_e + (1-ρ) * ∇p_e
        g²_{e+1} = ρ * g²_e + (1-ρ) * ∇p_e²
        p_{e+1} = p_e - lr * ∇p_e / √(g²_{e+1} - g_{e+1}²)

    This roughly corresponds to dividing the gradients by their standard deviation
    over the past batches, in a rolling-momentum fashion.
    The more "unstable" a gradient, the lower its effective learning-rate.
    
    """

    def __init__(self, lr, rho, eps=1e-7):
        df.Optimizer.__init__(self, lr=lr, rho=rho, eps=eps)

    def get_updates(self, params, grads, lr, rho, eps):
        updates = []

        for param, grad in zip(params, grads):
            g_state = df.utils.create_param_state_as(param)
            new_g = rho*g_state + (1-rho)*grad
            g2_state = df.utils.create_param_state_as(param)
            new_g2 = rho*g2_state+(1-rho)*grad*grad
            updates.append((g_state, new_g))
            updates.append((g2_state, new_g2))
            updates.append((param, param - lr*(grad/df.T.sqrt(new_g2-new_g*new_g+eps))))

        return updates
