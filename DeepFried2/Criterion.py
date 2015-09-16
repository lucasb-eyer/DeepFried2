import DeepFried2 as df


class Criterion:

    def __init__(self):
        self.penalties = []

    def symb_forward(self, symb_input, symb_target):
        raise NotImplementedError("`{}` needs to implement `symb_forward` method.".format(df.typename(self)))

    # TODO: Might actually want the weights to be shared variables so we can change their values on-the-fly!
    def add_penalty(self, weight_or_pen, pen=None):
        if pen is None:
            weight, pen = 1.0, weight_or_pen
        else:
            weight, pen = weight_or_pen, pen
        self.penalties.append((weight, pen))

    def full_symb_forward(self, symb_input, symb_target):
        cost = self.symb_forward(symb_input, symb_target)

        for w, p in self.penalties:
            cost += w*p.symb_forward()

        return cost
