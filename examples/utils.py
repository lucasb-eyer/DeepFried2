import sys as _sys
import numpy as _np


def printnow(fmt, *a, **kw):
    _sys.stdout.write(fmt.format(*a, **kw))
    _sys.stdout.flush()


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


# Reading and resizing images
#############################


try:
    import cv2 as _cv2

    def imread(fname, dtype=_np.float32):
        im = _cv2.imread(fname, flags=_cv2.IMREAD_UNCHANGED)
        if im is None:
            raise IOError("Couldn't open image file {}".format(fname))
        return im

    def imresize(im, h, w):
        if im.shape[:2] == (h, w):
            return im

        # Use AREA interpolation as soon as one dimension gets smaller to avoid moiree.
        inter = _cv2.INTER_AREA if im.shape[0] < h or im.shape[1] < w else _cv2.INTER_LINEAR
        return _cv2.resize(im, (w,h), interpolation=inter)

except ImportError:

    try:
        # This is what scipy's imread does lazily.
        from PIL import Image as _Image

        def imread(fname, dtype=_np.float32):
            # This does what CV_LOAD_IMAGE_ANYDEPTH does by default.
            return _np.array(_Image.open(fname), dtype=dtype)

        def imresize(im, h, w):
            if im.shape[:2] == (h, w):
                return im

            # This has problems re-reading a numpy array if it's not 8-bit anymore.
            assert im.max() > 1, "PIL has problems resizing images after they've been changed to e.g. [0-1] range. Either install OpenCV or resize right after reading the image."
            img = _Image.fromarray(im.astype(_np.uint8))
            return _np.array(img.resize((w,h), _Image.BILINEAR), dtype=im.dtype)

    except ImportError:

        def imread(fname, dtype=None):
            raise ImportError(
                "Neither OpenCV nor the Python Imaging Library (PIL) is "
                "installed. Please install either for loading images."
            )

        def imresize(im, h, w):
            raise ImportError(
                "Neither OpenCV nor the Python Imaging Library (PIL) is "
                "installed. Please install either for resizing images."
            )


def imresizecrop(img, size):
    assert not isinstance(size, tuple), "For now, `size` needs to be a single integer, i.e. it's being squared."

    # First, resize the smallest side to `size`.
    img = imresize(img, h=img.shape[0]*size//min(img.shape[:2]),
                        w=img.shape[1]*size//min(img.shape[:2]))

    # Then, crop-out the central part of the largest side.
    return img[(img.shape[0]-size)//2:img.shape[0]-(img.shape[0]-size)//2,
               (img.shape[1]-size)//2:img.shape[1]-(img.shape[1]-size)//2,
               :]
