"""M1/M2 消融臂（ABLATION_M1M2_PREREG_v1）：m1_pca / m1_direct / m2_gaussian。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scipy.linalg import helmert
from sklearn.covariance import OAS
from scipy.stats import multivariate_t, multivariate_normal

from bdh_pixel_loss_controls import conditional_likelihood as lik
from bdh_pixel_loss_controls.product_correlated_source import fixed_count
from bdh_pixel_loss_controls.few_label_transfer import utility_mass
from bdh_pixel_loss_controls.utility_transport import divide
from run_adaptive import (CACHE, project_class_evidence_rt, natural_policy, macro_ba,
                          src_weights, SEEDS, ORDINALS, DRAWS, K, TAU0)

OUT = Path(r'E:/Hermes/paper_tracking/sensitivity/novel_arms/ablations')
OUT.mkdir(parents=True, exist_ok=True)
EPSILON = 1e-12


def fit_source_gen(x, labels, ports):
    x, labels = np.asarray(x, dtype=np.float64), np.asarray(labels)
    assert x.shape == (len(labels), x.shape[1]) and np.isfinite(x).all()
    assert np.array_equal(np.unique(labels), np.arange(8))
    weights = src_weights(labels, ports)
    center = np.average(x, weights=weights, axis=0)
    scale = np.maximum(np.sqrt(np.average((x - center) ** 2, weights=weights, axis=0)), 1e-6)
    z = (x - center) / scale
    means = np.stack([np.average(z[labels == c], weights=weights[labels == c], axis=0) for c in range(8)])
    residual = (z - means[labels]) * np.sqrt(len(x) * weights[:, None])
    estimator = OAS(assume_centered=True).fit(residual)
    covariance = estimator.covariance_
    np.linalg.cholesky(covariance)
    return dict(center=center, scale=scale, means=means, covariance=covariance,
                precision=np.linalg.solve(covariance, np.eye(covariance.shape[0])),
                shrinkage=np.array(estimator.shrinkage_), weights=weights)


def predict_gaussian(source, annotation_x, labels, query_x):
    labels = np.asarray(labels)
    z = (np.asarray(annotation_x) - source['center']) / source['scale']
    query = (np.asarray(query_x) - source['center']) / source['scale']
    classes, d = source['means'].shape
    counts = np.bincount(labels, minlength=classes)
    sums = np.zeros((classes, d)); np.add.at(sums, labels, z)
    info = TAU0 + counts
    means = (TAU0 * source['means'] + sums) / info[:, None]
    within, shift = np.zeros((d, d)), np.zeros((d, d))
    for c in np.flatnonzero(counts):
        average = sums[c] / counts[c]
        residual = z[labels == c] - average
        delta = average - source['means'][c]
        within += residual.T @ residual
        shift += TAU0 * counts[c] / info[c] * np.outer(delta, delta)
    psi = TAU0 * source['covariance'] + within + shift
    nu = d + 1 + TAU0 + len(labels)
    shape_factors = (1 + 1 / info) / (nu - d + 1)
    loglik = np.column_stack([multivariate_normal.logpdf(query, mean=means[c],
        cov=psi * shape_factors[c]) for c in range(classes)])
    assert np.isfinite(loglik).all()
    return dict(counts=counts, means=means, loglikelihood=loglik)


def run(mode, smoke=False):
    out = {'ports': {}}
    for port in sorted(p.name[:-4] for p in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{port}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        base_logits = d['base_logits']; expert_logits = d['expert_logits']
        feats0 = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
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
            if mode == 'm1_pca':
                from sklearn.decomposition import PCA
                cat = np.c_[base_logits, expert_logits]
                pca = PCA(n_components=14, random_state=0).fit(cat[source_pos])
                feats = pca.transform(cat)
            elif mode == 'm1_direct':
                feats = np.c_[base_logits, expert_logits]
            else:
                feats = feats0
            source_model = fit_source_gen(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
            cond = np.bincount(K * K * y_ids[source_pos] + K * base[source_pos] + expert[source_pos],
                               weights=src_weights(y_ids[source_pos], ports_all[source_pos]),
                               minlength=K ** 3).reshape(K, K, K)
            cond = cond / cond.sum(axis=(1, 2))[:, None, None]
            for seed in SEEDS:
                for draw in range(DRAWS):
                    rng = np.random.default_rng(seed * 1000 + ordinal * 100 + draw)
                    ann = rng.choice(qpos, 64, replace=False)
                    ay = y_ids[ann]
                    observed = np.bincount(ay, minlength=K) > 0
                    priority = rng.permutation(query_n)
                    b, e = base[qpos], expert[qpos]
                    s_flat = divide(utility_mass(cond, observed, True),
                                    np.bincount(K * b + e, minlength=K * K).reshape(K, K) / len(b))[b, e]
                    if mode == 'm2_gaussian':
                        p1 = predict_gaussian(source_model, feats[ann], ay, feats[qpos])
                    else:
                        p1 = lik.predictive_likelihood if mode != 'm2_gaussian' else None
                        from bdh_pixel_loss_controls import shared_covariance_likelihood as shared_cov
                        p1 = shared_cov.predict(source_model, feats[ann], ay, feats[qpos])
                    f1 = lik.class_mass(p1['loglikelihood'], p1['counts'])
                    ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
                    s2 = lik.robust_scores(ratio2, b, e, observed)
                    arms = {'flat': s_flat, 'class_projection': s2}
                    natural = {name: natural_policy(b, e, s, ranking[qpos], priority, .05)
                               for name, s in arms.items()}
                    kk = min(int(natural[name]['selected'].sum()) for name in arms)
                    matched = {name: fixed_count(b, e, arms[name], ranking[qpos], priority, kk)
                               for name in arms}
                    y_q = y_ids[qpos]
                    for name, s in arms.items():
                        for mode_, mask in (('natural', natural[name]['selected']), ('matched', matched[name])):
                            pred = np.where(mask, expert[qpos], base[qpos])
                            records.append(dict(arm=name, mode=mode_, ba=macro_ba(y_q, pred)))
        rec = pd.DataFrame(records)
        per = rec.groupby(['arm', 'mode']).ba.agg('mean').reset_index()
        bf = per[(per['arm'] == 'flat') & (per['mode'] == 'matched')]['ba'].iloc[0]
        v = per[(per['arm'] == 'class_projection') & (per['mode'] == 'matched')]['ba'].iloc[0]
        nat = float(per[(per['arm'] == 'class_projection') & (per['mode'] == 'natural')]['ba'].iloc[0]
                    - per[(per['arm'] == 'flat') & (per['mode'] == 'natural')]['ba'].iloc[0])
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
    (OUT / f'summary_{mode}.json').write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    run(a.mode, smoke=a.smoke)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
