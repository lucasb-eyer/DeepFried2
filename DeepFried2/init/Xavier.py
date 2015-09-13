import numpy as _np

def xavier(gain=1):
    def init(shape, fan):
        assert fan is not None, "The parameter's `fan` needs to be specified when using Xavier initialization."

        fan_mean = _np.mean(fan)
        bound = gain * _np.sqrt(3./fan_mean)
        return _np.random.uniform(low=-bound, high=bound, size=shape)
    return init

def xavierN(gain=1):
    def init(shape, fan):
        assert fan is not None, "The parameter's `fan` needs to be specified when using Xavier initialization."

        fan_mean = _np.mean(fan)
        eff_std = gain * _np.sqrt(1./fan_mean)
        return eff_std * _np.random.randn(*shape)
    return init

def xavierSigm(gain=1):
    return xavier(gain * _np.sqrt(2))

def xavierSigmN(gain=1):
    return xavierN(gain * _np.sqrt(2))
