import DeepFried2 as df


class Concat(df.Container):
    def __init__(self, axis=1):
        df.Container.__init__(self)
        self.axis = axis

    def symb_forward(self, symb_inputs):
        assert isinstance(symb_inputs, (list, tuple)), "Input to `{}` container needs to be a tuple or a list.".format(df.utils.typename(self))
        return df.T.concatenate(symb_inputs, self.axis)
