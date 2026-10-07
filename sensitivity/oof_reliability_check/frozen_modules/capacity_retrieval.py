"""Pinned retrieve() extracted unchanged from c82933b; no unrelated imports."""
import numpy as np
from scipy.special import softmax
from ot.bregman import sinkhorn_log
from report_bdh_pixel_handoff import require

def retrieve(cost, capacity, mode):
    """Row attention versus fixed row AND prototype mass, at regularization1."""
    cost, capacity = np.asarray(cost, dtype=np.float64), np.asarray(capacity, dtype=np.float64)
    require(cost.ndim == 2 and cost.shape[0] > 0 and cost.shape[1] == len(capacity)
            and np.isfinite(cost).all() and np.isfinite(capacity).all() and (capacity > 0).all()
            and np.isclose(capacity.sum(), 1., atol=1e-12, rtol=1e-12), 'invalid evidence cost/capacity')
    n = len(cost)
    if mode == 'row':
        weights = softmax(np.log(capacity)[None] - cost, axis=1)
        diagnostics = dict(iterations=0)
    elif mode == 'capacity':
        # Row-constant shifts preserve the solution and improve numerical range.
        shifted = cost - cost.min(1, keepdims=True)
        coupling, log = sinkhorn_log(np.full(n, 1 / n), capacity, shifted, reg=1.,
                                    numItermax=2000, stopThr=1e-11, log=True)
        weights = n * coupling
        diagnostics = dict(iterations=int(log['niter']), log_u=log['log_u'], log_v=log['log_v'])
    else:
        raise ValueError('unknown evidence retrieval mode')
    row_error = float(np.abs(weights.sum(1) - 1).max())
    column_mass = weights.mean(0)
    column_error = float(np.abs(column_mass - capacity).max())
    require(np.isfinite(weights).all() and row_error <= 1e-8, 'invalid evidence row normalization')
    if mode == 'capacity':
        require(column_error <= 1e-8, 'evidence capacity solver did not converge')
    return dict(weights=weights, row_error=row_error, column_error=column_error,
                column_mass=column_mass, **diagnostics)


