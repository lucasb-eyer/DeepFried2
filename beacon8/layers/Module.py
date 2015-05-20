import theano as _th
import theano.tensor as _T


class Module:

    def __init__(self):
        self.training_mode = True

        self.fn_forward = None
        self.fn_accum_grads = None

    def reset(self):
        pass

    #def __hash__(self):
    #    raise NotImplementedError("You *need* to reimplement hash, even if it's just python's default. See the documentation for more info.")

    def zero_grad_parameters(self):
        _, grads = self.parameters()
        for grad in grads:
            grad.set_value(0 * grad.get_value())

    def parameters(self):
        params, grads = [], []

        if self.training_mode and hasattr(self, 'weight'):
            assert hasattr(self, 'grad_weight'), "The layer {} has a `weight` variable but no `grad_weight`, you probably forget to implement it.".format(type(self))
            params += [self.weight]
            grads += [self.grad_weight]

        if self.training_mode and hasattr(self, 'bias'):
            assert hasattr(self, 'grad_bias'), "The layer {} has a `bias` variable but no `grad_bias`, you probably forget to implement it.".format(type(self))
            params += [self.bias]
            grads += [self.grad_bias]

        return params, grads

    def evaluate(self):
        self.training_mode = False

    def training(self):
        self.training_mode = True

    def symb_forward(self, symb_input):
        raise NotImplementedError

    def forward(self, data):
        if self.fn_forward is None:
            symb_in = _T.TensorType(_th.config.floatX, (False,) * data.ndim)('X')
            symb_out = self.symb_forward(symb_in)
            self.fn_forward = _th.function(inputs=[symb_in], outputs=symb_out)

        return self.fn_forward(data)

    def accumulate_gradients(self, data_in, data_tgt, loss):
        if self.fn_accum_grads is None:
            symb_in = _T.TensorType(_th.config.floatX, (False,) * data_in.ndim)('X')
            symb_tgt = _T.TensorType(_th.config.floatX, (False,) * data_tgt.ndim)('T')
            symb_out = self.symb_forward(symb_in)
            symb_err = loss.symb_forward(symb_out, symb_tgt)

            params, grads = self.parameters()
            symb_grads = _th.grad(cost=symb_err, wrt=params)

            grads_updates = [(grad, grad + symb_grad) for grad, symb_grad in zip(grads, symb_grads)]
            self.fn_accum_grads = _th.function(
                inputs=[symb_in, symb_tgt],
                updates=grads_updates
            )

        self.fn_accum_grads(data_in, data_tgt)
