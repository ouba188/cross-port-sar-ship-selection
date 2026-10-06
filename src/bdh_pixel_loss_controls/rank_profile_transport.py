"""Mass-preserving source rank profiles; a task-specific candidate, not a guarantee."""

import numpy as np

from bdh_pixel_loss_controls import utility_transport, few_label_transfer
from report_bdh_pixel_handoff import source_weights, budget_policy, require, EPSILON


def rank_cells(base, expert, ranking, ports, priority, bins=4):
    """Descending within-port/predicted-pair midranks; no correctness labels."""
    pair = 8 * np.asarray(base) + np.asarray(expert)
    ranking, ports, priority = map(np.asarray, (ranking, ports, priority))
    cells = np.empty(len(pair), dtype=np.int64)
    for port in np.unique(ports):
        for group in np.unique(pair[ports == port]):
            use = np.flatnonzero((ports == port) & (pair == group))
            ordered = use[np.lexsort((priority[use], -ranking[use]))]
            cells[ordered] = np.floor(bins * (np.arange(len(use)) + .5) / len(use)).astype(np.int64)
    return cells


def fit_profile(labels, base, expert, ranking, ports, priority, bins=4, pseudocount=8.):
    """Conditional rank shapes shrunk to a flat density within each class/pair."""
    labels, base, expert, ports = map(np.asarray, (labels, base, expert, ports))
    require(bins > 0 and pseudocount > 0, 'positive resolution and shrinkage required')
    conditional = utility_transport.source_statistics(labels, base, expert, ports)['conditional']
    cells = rank_cells(base, expert, ranking, ports, priority, bins)
    weights = source_weights(labels, ports)
    group = 64 * labels + 8 * base + expert
    hist = np.bincount(bins * group + cells, weights=weights, minlength=512 * bins).reshape(8, 8, 8, bins)
    total = hist.sum(-1)
    squared = np.bincount(group, weights=weights ** 2, minlength=512).reshape(8, 8, 8)
    effective_n = np.divide(total ** 2, squared, out=np.zeros_like(total), where=squared > 0)
    fractions = np.divide(hist, total[..., None], out=np.full_like(hist, 1 / bins), where=total[..., None] > 0)
    density = bins * (effective_n[..., None] * fractions + pseudocount / bins) / (effective_n[..., None] + pseudocount)
    return dict(conditional=conditional, density=density, effective_n=effective_n,
                source_cells=cells, bins=np.array(bins), pseudocount=np.array(pseudocount))


def transport_mass(profile, base, expert, ranking, priority):
    """Preserve source class/pair mass while redistributing it over query rank cells."""
    base, expert = np.asarray(base), np.asarray(expert)
    bins = int(profile['bins'])
    pair = 8 * base + expert
    cells = rank_cells(base, expert, ranking, np.zeros(len(base), dtype=int), priority, bins)
    counts = np.bincount(bins * pair + cells, minlength=64 * bins).reshape(8, 8, bins)
    total = counts.sum(-1, keepdims=True)
    fractions = np.divide(counts, total, out=np.full(counts.shape, 1 / bins), where=total > 0)
    weighted = profile['density'] * fractions[None]
    normalized = weighted / weighted.sum(-1, keepdims=True)
    mass = profile['conditional'][..., None] * normalized
    return mass, counts / len(base), cells


def profile_scores(profile, base, expert, ranking, priority, observed):
    mass, query_mass, cells = transport_mass(profile, base, expert, ranking, priority)
    values = []
    for cell in range(int(profile['bins'])):
        numerator = few_label_transfer.utility_mass(mass[..., cell], observed, True)
        values.append(utility_transport.divide(numerator, query_mass[..., cell]))
    table = np.stack(values, axis=-1)
    return table[np.asarray(base), np.asarray(expert), cells]


def policy(base, expert, utility, ranking, priority, matched_k=None):
    """Positive utility first, then confidence; identical tie rule for both arms."""
    base, expert, utility, ranking, priority = map(np.asarray, (base, expert, utility, ranking, priority))
    # budget_policy retains the established cap and admission threshold.
    admitted = budget_policy(base, expert, utility, .05, priority)
    candidates = np.flatnonzero((base != expert) & (utility > EPSILON))
    order = candidates[np.lexsort((priority[candidates], -ranking[candidates], -utility[candidates]))]
    limit = int(admitted['selected'].sum())
    if matched_k is not None:
        require(0 <= matched_k <= limit, 'matched count exceeds natural count')
        limit = matched_k
    mask = np.zeros(len(base), dtype=bool)
    mask[order[:limit]] = True
    return dict(selected=mask, prediction=np.where(mask, expert, base), cap=admitted['cap'])
