import numpy as np
from collections import deque as _deque

from DeepFried2 import floatX


def mackey_glass(T, N, tau=17, n=10, beta=0.2, gamma=0.1, dt=10, mila=False):
    '''Returns `N` different Mackey Glass time-series of length `T` with delay `tau`.
    `tau` is the delay of the system, higher values being more chaotic.
    The values are centered and squashed through a tanh.

    dx/dt = beta * x_tau / (1 + x_tau ^ n) - gamma * x
    with x_tau = x(t - tau)

    The return shape is (N, T, 1).

    Origin of this function: https://github.com/mila-udem/summerschool2015
    but modified a slight bit (unless `mila` is True).
    '''
    X = np.empty((N, T, 1), floatX)
    x = 1.2  # Initial conditions for the history of the system

    for i in range(N):
        # Note that this is slightly different than the MILA one.
        # They didn't re-init x to 1.2 for each i, instead re-using the last
        # one from the previous i, all the while re-initializing the history.
        # I think what they do is wrong, but probably doesn't matter much.
        history = _deque((1.2 if mila else x) + beta * (np.random.rand(tau * dt) - 0.5))
        # TODO: Is x above really x or just 1.2 which they used in MILA one?
        # TODO: 0.5 above must be constructed from others in some way?

        for t in range(T):
            for _ in range(dt):
                # xtau is the value at the last timestep, dt ago.
                xtau = history.popleft()
                history.append(x)
                x += (beta * xtau / (1.0 + xtau**n) - gamma*x) / dt
            X[i,t,0] = x

    # Squash timeseries through tanh
    return np.tanh(X - 1)


def mso(T, N, freqs=[0.2, 0.311]):
    '''Returns `N` different sums of randomly phased sine-waves of
    frequencies `freqs`, each of length `T`.

    The return shape is (N, T, 1).

    Origin of this function: https://github.com/mila-udem/summerschool2015
    '''
    X = np.empty((N, T, 1), floatX)

    for i in range(N):
        phase = np.random.rand()
        X[i,:,0] = sum(np.sin(f*np.arange(T) + phase) for f in freqs)

    return X


def lorentz(T=1000, N=5, dt=0.01, sigma=10, rho=28, beta=8./3., x0=[0, -0.01, 9]):
    """Returns `N` independent Lorentz time-series of length `T` in (N,T,3).

    - `dt` is the time between samples. Currently, only a scalar is supported,
      but the plan is to support varying `dt`s.

    - All of `sigma`, `rho`, and `beta` can be either a single number shared
      across all `N` time-series, a 2-tuple meaning bounds of a uniform random
      interval, or an array of size `N`.

    - `x0` are initial conditions. Either 3 values (x0,y0,z0) can be specified
      for shared initial conditions across all `N`, or an `(N,3)` array for
      individual initial conditions.

    For any of `sigma`, `rho` and `beta`, a smaller value is "smoother" behaviour,
    but the difference from changing `x0` is much more dramatic then any of them.
    """
    if isinstance(sigma, tuple) and len(sigma) == 2:
        sigma = np.random.uniform(*sigma, size=N).astype(floatX)

    if isinstance(rho, tuple) and len(rho) == 2:
        rho = np.random.uniform(*rho, size=N).astype(floatX)

    if isinstance(beta, tuple) and len(beta) == 2:
        beta = np.random.uniform(*beta, size=N).astype(floatX)

    X = np.empty((N, T, 3), floatX)
    X[:,0] = x0  # Initial conditions taken from 'Chaos and Time Series Analysis', J. Sprott

    for t in range(T-1):
        X[:,t+1,0] = X[:,t,0] + sigma * (X[:,t,1] - X[:,t,0]) * dt
        X[:,t+1,1] = X[:,t,1] + (X[:,t,0] * (rho - X[:,t,2]) - X[:,t,1]) * dt
        X[:,t+1,2] = X[:,t,2] + (X[:,t,0] * X[:,t,1] - beta * X[:,t,2]) * dt

    return X
