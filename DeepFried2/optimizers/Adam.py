# -*- coding: utf-8 -*-
import DeepFried2 as df
import numpy as _np


class Adam(df.Optimizer):
    """
    Implements Diederik Kingma and Jimmy Ba's ADAM optimizer.
    The paper itself is really neat, and both very convincing and practical.

    TL;DR: 1. Handles grad. sparsity like RMSProp, moving target like AdaDelta.
           2. Has upper bounds on parameter updates! Roughly speaking,
              the bound is more-or-less `alpha` in typical cases. (Sec. 2.1)
           3. Corrects for initialization bias (Sec. 3)

    The updates are:

        m[e+1] = β1 * m[e] + (1-β1) * ∇p[e]
        v[e+1] = β2 * v[e] + (1-β2) * ∇p[e]²

        m[e+1] /= (1-β1)^t
        v[e+1] /= (1-β2)^t

        p[e+1] = p[e] - α * m[e+1] / (√v[e+1] + ε)

    Which can be re-organized for efficiency by merging the bias corrections
    into the α, since all of them are scalars.

    As in RMSProp, we need to add epsilons in order to create stability.
    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        df.Optimizer.__init__(self, alpha=alpha, beta1=beta1, beta2=beta2, eps=eps)

    def get_updates(self, params, grads, alpha, beta1, beta2, eps):
        updates = []


        #t_state = df.th.shared(_np.array(0, dtype=_np.int32), name='adam_t')
        t_state = df.th.shared(_np.float32(0), name='adam_t')
        t = t_state + 1
        eff_alpha = alpha * df.T.sqrt(1 - beta2**t)/(1 - beta1**t)

        for param, grad in zip(params, grads):
            m_state = df.utils.create_param_state_as(param, prefix='m_')
            v_state = df.utils.create_param_state_as(param, prefix='v_')

            new_m = beta1*m_state + (1-beta1)*grad
            new_v = beta2*v_state + (1-beta2)*grad**2

            up = eff_alpha * new_m / (df.T.sqrt(new_v) + eps)

            updates.append((param, param - up))
            updates.append((m_state, new_m))
            updates.append((v_state, new_v))

        updates.append((t_state, t))
        return updates
