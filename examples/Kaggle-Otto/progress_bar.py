from progressbar import ProgressBar, Counter, Percentage, Bar, ETA


def make_progressbar(mode, epoch, data_size):
    widgets = [mode + ' epoch #', str(epoch), ', processed ', Counter(), ' of ', str(data_size),
               ' (', Percentage(), ')', ' ', Bar(), ' ', ETA()]
    return ProgressBar(maxval=data_size, widgets=widgets)
