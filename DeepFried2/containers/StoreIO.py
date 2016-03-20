import DeepFried2 as df


class StoreIO(df.SingleModuleContainer):
    def __init__(self, module, inp=True, out=True):
        df.SingleModuleContainer.__init__(self, module)

        # We need to store them in dicts where the current mode is the key.
        # That's because we will have different instances in different modes.
        self._inp = {} if inp else None
        self._out = {} if out else None

    def symb_forward(self, symb_inp):
        symb_out = self.modules[0].symb_forward(symb_inp)

        if self._inp is not None:
            self._inp[self.training_mode] = symb_inp
        if self._out is not None:
            self._out[self.training_mode] = symb_out

        return symb_out

    def get_extra_outputs(self):
        return (df.utils.aslist(self._inp[self.training_mode], none_to_empty=True)
               +df.utils.aslist(self._out[self.training_mode], none_to_empty=True))

    @property
    def out(self):
        assert self._out is not None, "You want to look at the output you forbad to store?"

        _out = self._out[self.training_mode]
        if isinstance(_out, (list, tuple)):
            return [o.val for o in _out]
        else:
            return _out.val

    @property
    def inp(self):
        assert self._inp is not None, "You want to look at the input you forbad to store?"

        _inp = self._inp[self.training_mode]
        if isinstance(_inp, (list, tuple)):
            return [o.val for o in _inp]
        else:
            return _inp.val
