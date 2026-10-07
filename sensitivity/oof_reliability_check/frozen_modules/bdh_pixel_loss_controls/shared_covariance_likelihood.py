"""Shared covariance uncertainty with source-anchored class-specific means."""

import numpy as np
from scipy.stats import multivariate_t

from bdh_pixel_loss_controls.conditional_likelihood import TAU
from report_bdh_pixel_handoff import require


def predict(source, annotation_x, labels, query_x):
    labels = np.asarray(labels)
    z = (np.asarray(annotation_x) - source['center']) / source['scale']
    query = (np.asarray(query_x) - source['center']) / source['scale']
    classes, d = source['means'].shape
    require(labels.ndim == 1 and labels.dtype.kind in 'iu' and np.all((labels >= 0) & (labels < classes))
            and z.shape == (len(labels), d) and query.ndim == 2 and query.shape[1] == d
            and np.isfinite(z).all() and np.isfinite(query).all(), 'invalid covariance support/query')
    counts = np.bincount(labels, minlength=classes)
    sums = np.zeros((classes, d))
    np.add.at(sums, labels, z)
    information = TAU + counts
    means = (TAU * source['means'] + sums) / information[:, None]
    within, shift = np.zeros((d, d)), np.zeros((d, d))
    for c in np.flatnonzero(counts):
        average = sums[c] / counts[c]
        residual = z[labels == c] - average
        delta = average - source['means'][c]
        within += residual.T @ residual
        shift += TAU * counts[c] / information[c] * np.outer(delta, delta)
    psi = TAU * source['covariance'] + within + shift
    nu = d + 1 + TAU + len(labels)
    df = nu - d + 1
    shape_factors = (1 + 1 / information) / df
    loglikelihood = np.column_stack([multivariate_t.logpdf(query, loc=means[c],
        shape=psi * shape_factors[c], df=df) for c in range(classes)])
    require(np.isfinite(loglikelihood).all(), 'nonfinite shared-covariance predictive density')
    return dict(counts=counts, means=means, within_scatter=within, mean_shift_scatter=shift, psi=psi,
        nu=np.array(nu), df=np.array(df), information=information, shape_factors=shape_factors,
        covariance=psi / (nu - d - 1), loglikelihood=loglikelihood)

