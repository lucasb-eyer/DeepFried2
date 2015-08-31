import theano as th
import theano.tensor as T
floatX = th.config.floatX

from .layers import *
from .containers import *
from .criteria import *
from .optimizers import *
from . import zoo
from . import datasets
