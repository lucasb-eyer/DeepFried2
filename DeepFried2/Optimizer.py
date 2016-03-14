import DeepFried2 as df


class Optimizer(object):

    def __init__(self, **hyperparams):
        self.states = {}
        self.hyperparams = hyperparams

    def update_parameters(self, model):

        if model not in self.states:
            # TODO: Not only scalar, e.g. Adam might profit from integer t
            hyperparams = {name: df.T.scalar(name) for name in self.hyperparams}
            params, grads = zip(*[(p.param, p.grad) for p in model.parameters(trainable_only=True)])
            updates = self.get_updates(params, grads, **hyperparams)
            self.states[model] = df.th.function(
                inputs=list(hyperparams.values()),
                updates=updates
            )

        self.states[model](**self.hyperparams)

    def get_updates(self, params, grads):
        raise NotImplementedError("`{}` needs to implement `get_updates` method.".format(df.utils.typename(self)))

    def __repr__(self):
        return df.utils.typename(self) + "(" + ", ".join(k+"="+str(v) for k,v in self.hyperparams.items()) + ")"
