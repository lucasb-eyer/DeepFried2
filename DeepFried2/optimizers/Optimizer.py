import theano.tensor as _T
import theano as _th


class Optimizer:

    def __init__(self, **hyperparams):
        self.states = {}
        self.hyperparams = hyperparams

    def update_parameters(self, model):

        if model not in self.states:
            params, grads = model.parameters()
            # TODO: Not only scalar
            hyperparams = {name: _T.scalar(name) for name in self.hyperparams}
            updates = self.get_updates(params, grads, **hyperparams)
            self.states[model] = _th.function(
                inputs=list(hyperparams.values()),
                updates=updates
            )

        self.states[model](**self.hyperparams)

    def get_updates(self, params, grads):
        raise NotImplementedError
