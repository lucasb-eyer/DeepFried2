from ..optimizers import Optimizer


class SGD(Optimizer):

    def __init__(self, lr):
        Optimizer.__init__(self, lr=lr)

    def get_updates(self, params, grads, lr):
        return [(p, p - lr * g) for p, g in zip(params, grads)]
