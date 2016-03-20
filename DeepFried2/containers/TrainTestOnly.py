import DeepFried2 as df


class TrainOnly(df.SingleModuleContainer):
    """
    This only runs the contained module when we're in training mode.
    Note that it Doesn't change the mode of its contained module!

    TODO: Figure out whether this, or *always* changing the contained
          module to training mode makes more sense.
    """
    def symb_forward(self, symb_input):
        if self.training_mode:
            return self.modules[0].symb_forward(symb_input)
        else:
            return symb_input


class TestOnly(df.SingleModuleContainer):
    """
    This only runs the contained module when we're in evaluation mode.
    Note that it Doesn't change the mode of its contained module!

    TODO: Figure out whether this, or *always* changing the contained
          module to evaluation mode makes more sense.
    """
    def symb_forward(self, symb_input):
        if self.training_mode:
            return symb_input
        else:
            return self.modules[0].symb_forward(symb_input)
