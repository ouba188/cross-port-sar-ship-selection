"""Shared target class mass constrains cross-port prototype evidence retrieval.

POT solves standard entropy-regularized transport. This task-specific composition
is a development hypothesis, not a new transport solver or risk guarantee.
"""

import numpy as np
from ot.bregman import sinkhorn_log
from scipy.spatial.distance import cdist
from scipy.special import softmax

import bdh_shared_geometry_handoff as baseline
from bdh_pixel_loss_controls import conditional_likelihood as likelihood
from bdh_pixel_loss_controls.action_prototype_attention import make_bank
from report_bdh_pixel_handoff import require


def within_class_supply(labels, ports, counts):
    """Equal mass per present port, empirical action frequencies within class."""
    labels, ports, counts = map(np.asarray, (labels, ports, counts))
    supply = np.zeros(len(labels), dtype=np.float64)
    for label in np.unique(labels):
        present = np.unique(ports[labels == label])
        for port in present:
            use = (labels == label) & (ports == port)
            supply[use] = counts[use] / counts[use].sum() / len(present)
    return supply


def build_bank(source, features, labels, base, expert, ports):
    """Input rows must already obey source/target port and identity isolation."""
    require(np.array_equal(np.unique(labels), np.arange(len(source['means']))),
            'this fixed prototype method requires all source classes')
    z = (features - source['center']) / source['scale']
    bank = make_bank(z, labels, base, expert, ports)
    supply = within_class_supply(bank['labels'], bank['ports'], bank['counts'])
    centroids = np.zeros_like(source['means'], dtype=np.float64)
    np.add.at(centroids, bank['labels'], supply[:, None] * bank['features'])
    return dict(bank, within_class_supply=supply,
                centered_features=bank['features'].astype(np.float64) - centroids[bank['labels']])


def evidence_cost(source, adapted, bank, query_features):
    """Same archived predictive-t metric for both retrieval modes; no refitting."""
    query = (query_features - source['center']) / source['scale']
    anchored = bank['centered_features'] + adapted['means'][bank['labels']]
    factor = np.linalg.cholesky(adapted['covariance'])
    q, k = np.linalg.solve(factor, query.T).T, np.linalg.solve(factor, anchored.T).T
    distances = cdist(q, k, metric='sqeuclidean')
    d, df = query.shape[1], float(adapted['df'])
    shape_ratio = adapted['shape_factors'][bank['labels']] * (float(adapted['nu']) - d - 1)
    return .5 * d * np.log(shape_ratio)[None] + .5 * (df + d) * np.log1p(
        distances / (df * shape_ratio[None]))


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


def class_evidence(weights, labels, prior):
    """Model-based posterior and class-balanced ratios, not target-truth risks."""
    posterior = weights @ np.eye(len(prior))[labels]
    return dict(posterior=posterior, ratio=posterior / prior[None])


def predict_handoffs(source, bank, annotation_base_logits, annotation_expert_logits, annotation_labels,
                     query_base_logits, query_expert_logits, priority=None):
    """Connect the candidate four stages while retaining the baseline explicitly.

    Source rows must be isolated before constructing the bank. This interface
    requires the original64 target labels and provides no risk certificate.
    """
    original = baseline.predict(source, annotation_base_logits, annotation_expert_logits, annotation_labels,
                                query_base_logits, query_expert_logits, priority)
    features = likelihood.decision_features(query_base_logits, query_expert_logits)
    cost = evidence_cost(source, original['geometry'], bank, features)
    prior = original['evidence']['prior']
    capacity = prior[bank['labels']] * bank['within_class_supply']
    retrieved = retrieve(cost, capacity, 'capacity')
    evidence = class_evidence(retrieved.pop('weights'), bank['labels'], prior)
    evidence['utility'] = likelihood.robust_scores(evidence['ratio'], original['decision']['base'],
                                                   original['decision']['expert'], original['evidence']['observed'])
    decision = baseline.select_budgeted_handoffs(query_base_logits, query_expert_logits, evidence['utility'], priority)
    return dict(baseline=original, evidence=evidence, capacity=capacity, retrieval=retrieved, decision=decision)
