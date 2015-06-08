import sys as _sys


# Progressbar
##############


try:
    import progressbar as _pb

    def make_progressbar(prefix, data_size):
        widgets = [prefix, ', processed ', _pb.Counter(), ' of ', str(data_size),
                   ' (', _pb.Percentage(), ')', ' ', _pb.Bar(), ' ', _pb.ETA()]
        return _pb.ProgressBar(maxval=data_size, widgets=widgets)

except ImportError:

    class SimpleProgressBar(object):
        def __init__(self, tot, fmt):
            self.tot = tot
            self.fmt = fmt

        def start(self):
            self.update(0)

        def update(self, i):
            _sys.stdout.write("\r" + self.fmt.format(i=i, tot=self.tot, pct=float(i)/self.tot))
            _sys.stdout.flush()
            self.lasti = i

        def finish(self):
            _sys.stdout.write("\r" + self.fmt.format(i=self.lasti, tot=self.tot, pct=1.0) + "\n")
            _sys.stdout.flush()

    def make_progressbar(prefix, data_size):
        return SimpleProgressBar(data_size, prefix + ", processed {i} of {tot} ({pct:.2%})")
