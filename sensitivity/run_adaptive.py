"""自适应变体实验（ADAPTIVE_TRIGGER_PREREG_v1）。冻结缓存复用；与 sweep 同协议。
用法：python run_adaptive.py --mode adaptive_tau|adaptive_pseudo|adaptive_budget|adaptive_annot [--smoke]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from bdh_pixel_loss_controls import conditional_likelihood as lik
from bdh_pixel_loss_controls import shared_covariance_likelihood as shared_cov
from bdh_pixel_loss_controls.product_correlated_source import fixed_count
from bdh_capacity_evidence import retrieve
from bdh_pixel_loss_controls.few_label_transfer import utility_mass
from bdh_pixel_loss_controls.utility_transport import divide
from report_bdh_pixel_handoff import budget_policy
from scipy.stats import multivariate_t

WS = Path(r'E:/临时会话/visual_reliable_baseline')
OUT = Path(r'E:/Hermes/paper_tracking/sensitivity')
CACHE = OUT / 'pred_cache'
AD = OUT / 'adaptive'
AD.mkdir(parents=True, exist_ok=True)
SEEDS = (20261005, 20261006)
ORDINALS = (0, 1)
DRAWS = 20
K = 8
EPSILON = 1e-12
TAU0 = 8.


def macro_ba(y, pred):
    present = np.unique(y)
    return float(np.mean([(pred[y == c] == c).mean() for c in present]))


def natural_policy(base, expert, utility, ranking, priority, frac):
    base, expert, utility, ranking, priority = map(np.asarray, (base, expert, utility, ranking, priority))
    admitted = budget_policy(base, expert, utility, frac, priority)
    candidates = np.flatnonzero((base != expert) & (utility > EPSILON))
    order = candidates[np.lexsort((priority[candidates], -ranking[candidates], -utility[candidates]))]
    limit = int(admitted['selected'].sum())
    mask = np.zeros(len(base), dtype=bool)
    mask[order[:limit]] = True
    return dict(selected=mask, cap=admitted['cap'])


def project_class_evidence_rt(loglikelihood, prior):
    cost = -np.asarray(loglikelihood, dtype=np.float64)
    try:
        retrieval = retrieve(cost, prior, 'capacity')
    except ValueError:
        import ot
        shifted = cost - cost.min(1, keepdims=True)
        coupling = ot.sinkhorn(np.full(len(cost), 1 / len(cost)), prior, shifted,
                               reg=1., numItermax=50000, stopThr=1e-11)
        weights = len(cost) * coupling
        if float(np.abs(weights.mean(0) - prior).max()) > 1e-8:
            raise RuntimeError('solver failed')
        retrieval = dict(weights=weights)
    return retrieval['weights'] / prior[None]


def shared_cov_predict_tauvec(source, annotation_x, labels, query_x, tau_vec):
    """shared_cov.predict 的逐类 tau 版本；tau_vec==8 时与冻结版逐位一致（已验证）。"""
    labels = np.asarray(labels)
    z = (np.asarray(annotation_x) - source['center']) / source['scale']
    query = (np.asarray(query_x) - source['center']) / source['scale']
    classes, d = source['means'].shape
    counts = np.bincount(labels, minlength=classes)
    sums = np.zeros((classes, d))
    np.add.at(sums, labels, z)
    information = tau_vec + counts
    means = (tau_vec[:, None] * source['means'] + sums) / information[:, None]
    within, shift = np.zeros((d, d)), np.zeros((d, d))
    for c in np.flatnonzero(counts):
        average = sums[c] / counts[c]
        residual = z[labels == c] - average
        delta = average - source['means'][c]
        within += residual.T @ residual
        shift += tau_vec[c] * counts[c] / information[c] * np.outer(delta, delta)
    psi = TAU0 * source['covariance'] + within + shift
    nu = d + 1 + TAU0 + len(labels)
    df = nu - d + 1
    shape_factors = (1 + 1 / information) / df
    loglikelihood = np.column_stack([multivariate_t.logpdf(query, loc=means[c],
        shape=psi * shape_factors[c], df=df) for c in range(classes)])
    assert np.isfinite(loglikelihood).all()
    return dict(counts=counts, means=means, loglikelihood=loglikelihood)


def class_mass_pseudovec(loglikelihood, counts, pseudo_vec):
    from scipy.special import logsumexp
    pseudo = counts + pseudo_vec
    prior = pseudo / pseudo.sum()
    for _ in range(lik.EM_STEPS + 1):
        joint = loglikelihood + np.log(prior)
        marginal = logsumexp(joint, axis=1)
        posterior = np.exp(joint - marginal[:, None])
        if _ < lik.EM_STEPS:
            prior = (posterior.sum(axis=0) + pseudo) / (len(posterior) + pseudo.sum())
    return dict(prior=prior, posterior=posterior, ratio=posterior / prior)


def src_weights(labels, ports):
    result = np.empty(len(labels), dtype=np.float64)
    for port in np.unique(ports):
        mask = ports == port
        classes, inverse, counts = np.unique(labels[mask], return_inverse=True, return_counts=True)
        result[mask] = 1. / (len(np.unique(ports)) * len(classes) * counts[inverse])
    return result


def run(mode, smoke=False):
    out = {'ports': {}}
    for port in sorted(p.name[:-4] for p in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{port}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            out['ports'][port] = {'n': int(d['n']), 'skipped_small': True}
            continue
        base_logits = d['base_logits']; expert_logits = d['expert_logits']
        feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
        qpos = d['qpos']; spos = d['spos']; ranking = d['ranking']
        base = base_logits.argmax(1); expert = expert_logits.argmax(1)
        query_n = len(qpos)
        source_n = min(query_n, len(spos))
        records = []
        for ordinal in ORDINALS:
            rng_s = np.random.default_rng(1000 + ordinal)
            strat = {}
            for c in np.unique(y_ids[spos]):
                pos = spos[y_ids[spos] == c]
                strat[c] = rng_s.choice(pos, max(1, int(round(source_n * len(pos) / len(spos)))), replace=False)
            source_pos = np.concatenate(list(strat.values()))
            source_model = lik.fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
            cond = np.bincount(K * K * y_ids[source_pos] + K * base[source_pos] + expert[source_pos],
                               weights=src_weights(y_ids[source_pos], ports_all[source_pos]),
                               minlength=K ** 3).reshape(K, K, K)
            cond = cond / cond.sum(axis=(1, 2))[:, None, None]
            # 自适应 τ：按类源样本数
            if mode == 'adaptive_tau':
                src_counts = np.bincount(y_ids[source_pos], minlength=K).astype(float)
                med = np.median(src_counts[src_counts > 0])
                tau_vec = np.clip(TAU0 * med / np.maximum(src_counts, 1), 1., 64.)
            else:
                tau_vec = np.full(K, TAU0)
            # 自适应伪计数：Dirichlet 收缩按源先验
            if mode == 'adaptive_pseudo':
                src_freq = np.bincount(y_ids[source_pos], minlength=K).astype(float)
                src_freq = src_freq / src_freq.sum()
                pseudo_vec = 0.5 * (src_freq * 8 + 1)
            else:
                pseudo_vec = np.full(K, 0.5)
            for seed in SEEDS:
                for draw in range(DRAWS):
                    rng = np.random.default_rng(seed * 1000 + ordinal * 100 + draw)
                    if query_n <= 64:
                        continue
                    if mode == 'adaptive_annot':
                        q_ent = -(np.exp(base_logits[qpos] - base_logits[qpos].max(1, keepdims=True)));
                        q_ent = np.exp(q_ent); q_ent = q_ent / q_ent.sum(1, keepdims=True)
                        ent = -(q_ent * np.log(q_ent + 1e-12)).sum(1)
                        ann = qpos[np.argsort(-ent)[:64]]
                    else:
                        ann = rng.choice(qpos, 64, replace=False)
                    ay = y_ids[ann]
                    observed = np.bincount(ay, minlength=K) > 0
                    priority = rng.permutation(query_n)
                    b, e = base[qpos], expert[qpos]
                    s_flat = divide(utility_mass(cond, observed, True),
                                    np.bincount(K * b + e, minlength=K * K).reshape(K, K) / len(b))[b, e]
                    p1 = shared_cov_predict_tauvec(source_model, feats[ann], ay, feats[qpos], tau_vec)
                    f1 = class_mass_pseudovec(p1['loglikelihood'], p1['counts'], pseudo_vec)
                    s1 = lik.robust_scores(f1['ratio'], b, e, observed)
                    ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
                    s2 = lik.robust_scores(ratio2, b, e, observed)
                    arms = {'flat': s_flat, 'covariance_likelihood': s1, 'class_projection': s2}
                    if mode == 'adaptive_budget':
                        pos_util = float(np.mean(s2 > 0))
                        k = int(np.clip(round(query_n * pos_util), 0.02 * query_n, 0.10 * query_n))
                        natural = {arm: natural_policy(b, e, s, ranking[qpos], priority, k / query_n)
                                   for arm, s in arms.items()}
                    else:
                        natural = {arm: natural_policy(b, e, s, ranking[qpos], priority, .05)
                                   for arm, s in arms.items()}
                    kk = min(int(v['selected'].sum()) for v in natural.values())
                    matched = {arm: fixed_count(b, e, s, ranking[qpos], priority, kk) for arm, s in arms.items()}
                    y_q = y_ids[qpos]
                    for arm, s in arms.items():
                        for mode_, mask in (('natural', natural[arm]['selected']), ('matched', matched[arm])):
                            pred = np.where(mask, expert[qpos], base[qpos])
                            records.append(dict(arm=arm, mode=mode_, ba=macro_ba(y_q, pred)))
        rec = pd.DataFrame(records)
        per = rec.groupby(['arm', 'mode']).ba.agg(['mean', 'std']).reset_index()
        bf = per[(per['arm'] == 'flat') & (per['mode'] == 'matched')]['mean'].iloc[0]
        v = per[(per['arm'] == 'class_projection') & (per['mode'] == 'matched')]['mean'].iloc[0]
        nat = float(per[(per['arm'] == 'class_projection') & (per['mode'] == 'natural')]['mean'].iloc[0]
                    - per[(per['arm'] == 'flat') & (per['mode'] == 'natural')]['mean'].iloc[0])
        out['ports'][port] = {'n': int(d['n']), 'cp_flat': v - bf, 'natural': nat}
        print(f'{mode} {port}: {100*(v-bf):+.3f}pp', flush=True)
        if smoke:
            break
    plist = [p for p in out['ports'] if 'cp_flat' in out['ports'][p]]
    cp = np.array([out['ports'][p]['cp_flat'] for p in plist])
    nat = np.array([out['ports'][p]['natural'] for p in plist])
    res = {'mode': mode, 'n_ports': len(plist), 'cp_flat_mean': float(cp.mean()),
           'cp_flat_positive': int((cp > 0).sum()), 'cp_flat_worst': float(cp.min()),
           'natural_mean': float(nat.mean())}
    (AD / f'summary_{mode}.json').write_text(
        __import__('json').dumps(res, ensure_ascii=False), encoding='utf-8')
    print(__import__('json').dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    run(a.mode, smoke=a.smoke)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
