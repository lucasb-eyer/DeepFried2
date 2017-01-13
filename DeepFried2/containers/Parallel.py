import DeepFried2 as df


class Parallel(df.Container):
    def symb_forward(self, symb_input):
        assert isinstance(symb_input, (list, tuple)), "`{}` must have >1 inputs".format(df.utils.typename(self))
        assert len(symb_input) == len(self.modules), "`{}` should have the same number of inputs ({}) as modules ({}).".format(df.utils.typename(self), len(symb_input), len(self.modules))
        return tuple(module(symb_in) for module, symb_in in zip(self.modules, symb_input))
