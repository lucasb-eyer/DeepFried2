import DeepFried2 as df


class Optimizer:

    def __init__(self, **hyperparams):
        self.states = {}
        self.hyperparams = hyperparams

    def update_parameters(self, model):

        if model not in self.states:
            params, grads = model.unique_parameters()
            # TODO: Not only scalar, e.g. Adam might profit from integer t
            hyperparams = {name: df.T.scalar(name) for name in self.hyperparams}
            updates = self.get_updates(params, grads, **hyperparams)
            self.states[model] = df.th.function(
                inputs=list(hyperparams.values()),
                updates=updates
            )

        self.states[model](**self.hyperparams)

    def get_updates(self, params, grads):
        raise NotImplementedError

    def __repr__(self):
        return type(self).__name__ + "(" + ", ".join(k+"="+str(v) for k,v in self.hyperparams.items()) + ")"
