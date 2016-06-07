import DeepFried2 as df


class ActiveIn(df.SingleModuleContainer):
    """
    This only runs the contained module when we're in a specific mode.
    Note that it doesn't change the mode of its contained module!

    TODO: Figure out whether this, or *always* changing the contained
          module to training mode makes more sense.
    """
    def __init__(self, mode, module):
        df.SingleModuleContainer.__init__(self, module)
        self.active_mode = mode

    def symb_forward(self, symb_input):
        if self._mode == self.active_mode:
            return self.modules[0](symb_input)
        else:
            return symb_input


class InactiveIn(df.SingleModuleContainer):
    """
    This only runs the contained module when we're _not_ in a specific mode.
    Note that it doesn't change the mode of its contained module!

    TODO: Figure out whether this, or *always* changing the contained
          module to training mode makes more sense.
    """
    def __init__(self, mode, module):
        df.SingleModuleContainer.__init__(self, module)
        self.inactive_mode = mode

    def symb_forward(self, symb_input):
        if self._mode != self.inactive_mode:
            return self.modules[0](symb_input)
        else:
            return symb_input


class TrainingOnly(ActiveIn):
    """ For compatibility: Run contained module only if in training mode. """
    def __init__(self, module):
        ActiveIn.__init__(self, 'train', module)

class TestingOnly(InactiveIn):
    """ For compatibility: Run contained module only when _not_ in training mode. """
    def __init__(self, module):
        InactiveIn.__init__(self, 'train', module)
