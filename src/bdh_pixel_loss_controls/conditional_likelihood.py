"""Source-anchored continuous decision likelihoods for balanced handoff utility."""

import numpy as np
from scipy.linalg import helmert
from scipy.special import logsumexp
from sklearn.covariance import OAS

from report_bdh_pixel_handoff import require, source_weights


TAU = 8.
EM_STEPS = 128
PSEUDOCOUNT = .5


def decision_features(base_logits, expert_logits):
    arrays = [np.asarray(v, dtype=np.float64) for v in (base_logits, expert_logits)]
    require(arrays[0].ndim == 2 and arrays[0].shape[1] == 8
            and arrays[1].shape == arrays[0].shape
            and all(np.isfinite(v).all() for v in arrays), 'invalid paired logits')
    contrast = helmert(8, full=False)
    return np.concatenate([v @ contrast.T for v in arrays], axis=1)


def fit_source(x, labels, ports):
    x, labels = np.asarray(x, dtype=np.float64), np.asarray(labels)
    require(x.shape == (len(labels), 14) and np.isfinite(x).all()
            and np.array_equal(np.unique(labels), np.arange(8)), 'source needs 14 features and eight classes')
    weights = source_weights(labels, ports)
    center = np.average(x, weights=weights, axis=0)
    scale = np.maximum(np.sqrt(np.average((x - center) ** 2, weights=weights, axis=0)), 1e-6)
    z = (x - center) / scale
    means = np.stack([np.average(z[labels == c], weights=weights[labels == c], axis=0)
                      for c in range(8)])
    # OAS is applied to balanced residual rows; its iid shrinkage theory is not asserted here.
    residual = (z - means[labels]) * np.sqrt(len(x) * weights[:, None])
    estimator = OAS(assume_centered=True).fit(residual)
    covariance = estimator.covariance_
    np.linalg.cholesky(covariance)
    return dict(center=center, scale=scale, means=means, covariance=covariance,
                precision=np.linalg.solve(covariance, np.eye(14)),
                shrinkage=np.array(estimator.shrinkage_), weights=weights)


def predictive_likelihood(source, annotation_x, annotation_labels, query_x, adapt):
    labels = np.asarray(annotation_labels)
    annotation_x, query_x = np.asarray(annotation_x), np.asarray(query_x)
    require(labels.ndim == 1 and labels.dtype.kind in 'iu' and len(labels) > 0
            and np.all((labels >= 0) & (labels < 8)), 'invalid support labels')
    require(annotation_x.shape == (len(labels), 14) and query_x.ndim == 2
            and query_x.shape[1] == 14 and len(query_x) > 0
            and np.isfinite(annotation_x).all() and np.isfinite(query_x).all(), 'invalid support/query features')
    counts = np.bincount(labels, minlength=8)
    z = (annotation_x - source['center']) / source['scale']
    means = source['means'].copy()
    information = counts if adapt else np.zeros(8, dtype=int)
    if adapt:
        sums = np.zeros((8, 14))
        np.add.at(sums, labels, z)
        means = (TAU * means + sums) / (TAU + counts[:, None])
    inflation = 1. + 1. / (TAU + information)
    qz = (query_x - source['center']) / source['scale']
    delta = qz[:, None, :] - means[None]
    squared = np.einsum('ncd,de,nce->nc', delta, source['precision'], delta, optimize=True)
    logdet = np.linalg.slogdet(source['covariance'])[1]
    loglikelihood = -.5 * (squared / inflation + 14 * np.log(inflation)
                           + logdet + 14 * np.log(2 * np.pi))
    require(np.isfinite(loglikelihood).all(), 'nonfinite predictive likelihood')
    return dict(loglikelihood=loglikelihood, means=means, inflation=inflation, counts=counts)


def class_mass(loglikelihood, counts, query_weight=1.):
    """Fixed EM iteration budget; count-anchored mixture, not an optimality claim."""
    loglikelihood, counts = np.asarray(loglikelihood), np.asarray(counts)
    require(loglikelihood.ndim == 2 and loglikelihood.shape[1] == 8
            and len(loglikelihood) > 0 and np.isfinite(loglikelihood).all()
            and counts.shape == (8,) and np.all(counts >= 0), 'invalid mixture inputs')
    query_weight = float(query_weight)
    require(np.isfinite(query_weight) and 0 <= query_weight <= 1, 'invalid query likelihood weight')
    pseudo = counts + PSEUDOCOUNT
    prior = pseudo / pseudo.sum()
    curve = []
    for step in range(EM_STEPS + 1):
        joint = loglikelihood + np.log(prior)
        marginal = logsumexp(joint, axis=1)
        posterior = np.exp(joint - marginal[:, None])
        curve.append(float(query_weight * marginal.sum() + pseudo @ np.log(prior)))
        if step < EM_STEPS:
            prior = (query_weight * posterior.sum(axis=0) + pseudo) / (query_weight * len(posterior) + pseudo.sum())
    curve = np.asarray(curve)
    require(np.min(np.diff(curve)) >= -1e-9 * max(1., abs(curve[0])), 'mixture objective decreased')
    next_prior = (query_weight * posterior.sum(axis=0) + pseudo) / (query_weight * len(posterior) + pseudo.sum())
    return dict(prior=prior, posterior=posterior, ratio=posterior / prior,
                objective=curve, fixed_point_error=np.array(np.max(np.abs(prior - next_prior))))


def robust_scores(ratio, base, expert, observed):
    ratio, base, expert, observed = map(np.asarray, (ratio, base, expert, observed))
    require(ratio.shape == (len(base), 8) and base.shape == expert.shape
            and np.isfinite(ratio).all() and np.all(ratio >= 0)
            and observed.shape == (8,) and observed.dtype.kind == 'b' and observed.any(), 'invalid utility inputs')
    supports = ((np.arange(1, 256)[:, None] >> np.arange(8)) & 1).astype(bool)
    supports = supports[np.all(supports[:, observed], axis=1)]
    r = ratio[np.arange(len(base)), expert]
    h = ratio[np.arange(len(base)), base]
    values = (supports[:, expert] * r - supports[:, base] * h) / supports.sum(axis=1)[:, None]
    return values.min(axis=0)
