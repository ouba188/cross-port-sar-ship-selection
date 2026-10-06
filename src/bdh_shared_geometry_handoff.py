"""Runnable four-stage method assembled from the verified shared-geometry route.

This entry point accepts frozen predictor logits and never reads query labels.
The caller owns source/support/query identity and acquisition separation.
"""

import numpy as np

from bdh_pixel_loss_controls import conditional_likelihood as likelihood
from bdh_pixel_loss_controls import shared_covariance_likelihood as geometry
from bdh_pixel_loss_controls.pair_resolution_batch import simple_scores
from bdh_pixel_loss_controls.rank_profile_transport import policy
from report_bdh_pixel_handoff import require


ANNOTATION_BUDGET = 64


def build_cross_port_prior(base_logits, expert_logits, labels, ports):
    """M1: equal-port/class source moments in paired decision coordinates."""
    features = likelihood.decision_features(base_logits, expert_logits)
    return likelihood.fit_source(features, labels, ports)


def adapt_target_geometry(source, annotation_base_logits, annotation_expert_logits,
                          annotation_labels, query_base_logits, query_expert_logits):
    """M2: same64-label class means and shared covariance predictive evidence."""
    labels = np.asarray(annotation_labels)
    require(labels.shape == (ANNOTATION_BUDGET,), 'this frozen method requires exactly 64 annotation labels')
    support = likelihood.decision_features(annotation_base_logits, annotation_expert_logits)
    query = likelihood.decision_features(query_base_logits, query_expert_logits)
    return geometry.predict(source, support, labels, query)


def infer_handoff_utility(adapted, base, expert):
    """M3: unlabeled query mixture and support-constrained balanced utility."""
    fitted = likelihood.class_mass(adapted['loglikelihood'], adapted['counts'])
    observed = adapted['counts'] > 0
    utility = likelihood.robust_scores(fitted['ratio'], base, expert, observed)
    return dict(fitted, observed=observed, utility=utility)


def select_budgeted_handoffs(base_logits, expert_logits, utility, priority=None):
    """M4: positive disagreements under the unchanged 5% replacement cap."""
    base, expert = np.asarray(base_logits).argmax(1), np.asarray(expert_logits).argmax(1)
    priority = np.arange(len(base)) if priority is None else np.asarray(priority)
    require(priority.shape == base.shape, 'query priority length differs')
    _, ranking = simple_scores(base_logits, expert_logits)
    selected = policy(base, expert, utility, ranking, priority)
    return dict(selected, base=base, expert=expert, ranking=ranking, priority=priority)


def predict(source, annotation_base_logits, annotation_expert_logits, annotation_labels,
            query_base_logits, query_expert_logits, priority=None):
    """Apply M2-M4 using a fitted or archived M1 prior; no source refit."""
    adapted = adapt_target_geometry(source, annotation_base_logits, annotation_expert_logits,
                                    annotation_labels, query_base_logits, query_expert_logits)
    base = np.asarray(query_base_logits).argmax(1)
    expert = np.asarray(query_expert_logits).argmax(1)
    evidence = infer_handoff_utility(adapted, base, expert)
    decision = select_budgeted_handoffs(query_base_logits, query_expert_logits, evidence['utility'], priority)
    return dict(geometry=adapted, evidence=evidence, decision=decision)
