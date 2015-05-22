from .Optimizer import Optimizer
from ..utils import create_param_state_as


class Momentum(Optimizer):

    def __init__(self, lr, momentum):
        Optimizer.__init__(self, lr=lr, momentum=momentum)

    def get_updates(self, params, grads, lr, momentum):
        updates = []

        for param, grad in zip(params, grads):
            param_mom = create_param_state_as(param)
            v = momentum * param_mom - lr * grad
            updates.append((param_mom, v))
            updates.append((param, param + v))

        return updates
