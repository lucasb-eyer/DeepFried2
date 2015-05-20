from .optimizers import Optimizer


class SGD(Optimizer):

    def __init__(self, lr):
        super().__init__(lr=lr)

    def get_updates(self, params, grads, lr):
        return [g - lr * p for g, p in zip(grads, params)]
