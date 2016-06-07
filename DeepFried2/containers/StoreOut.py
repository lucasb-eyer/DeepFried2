import DeepFried2 as df


class StoreOut(df.SingleModuleContainer):

    def get_extra_outputs(self):
        return df.utils.flatten(self._last_symb_out.get(self._mode), none_to_empty=True)

    @property
    def out(self):
        _out = self._last_symb_out.get(self._mode)
        if _out is None:
            return None
        elif isinstance(_out, (list, tuple)):
            return [o.val for o in _out]
        else:
            return _out.val
