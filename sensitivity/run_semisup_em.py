"""半监督 EM 环（SEMISUP_EM_PREREG_v1）：伪目标均值回灌 M2，EM 1 轮。
用法：python run_semisup_em.py --lam 0|0.5|1 [--smoke]（λ=0 复现臂）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scipy.stats import multivariate_t

from bdh_pixel_loss_controls import conditional_likelihood as lik
from bdh_pixel_loss_controls.product_correlated_source import fixed_count
from bdh_pixel_loss_controls.few_label_transfer import utility_mass
from bdh_pixel_loss_controls.utility_transport import divide
from run_adaptive import (CACHE, project_class_evidence_rt, natural_policy, macro_ba,
                          src_weights, SEEDS, ORDINALS, DRAWS, K, TAU0)

OUT = Path(r'E:/Hermes/paper_tracking/sensitivity/semisup_em')
OUT.mkdir(parents=True, exist_ok=True)


def predict_with_pseudo(source, annotation_x, labels, query_x, pseudo_means, pseudo_n, lam):
    """shared_cov.predict 的回灌参数化副本：pseudo 项按 τ 路径入均值和协方差。"""
    labels = np.asarray(labels)
    z = (np.asarray(annotation_x) - source['center']) / source['scale']
    query = (np.asarray(query_x) - source['center']) / source['scale']
    classes, d = source['means'].shape
    counts = np.bincount(labels, minlength=classes)
    sums = np.zeros((classes, d))
    np.add.at(sums, labels, z)
    tau = np.full(classes, TAU0)
    info = tau + counts + lam * pseudo_n
    means = (tau[:, None] * source['means'] + sums + lam * pseudo_n[:, None] * pseudo_means) / info[:, None]
    within, shift = np.zeros((d, d)), np.zeros((d, d))
    for c in np.flatnonzero(counts):
        average = sums[c] / counts[c]
        residual = z[labels == c] - average
        delta = average - source['means'][c]
        within += residual.T @ residual
        shift += tau[c] * counts[c] / info[c] * np.outer(delta, delta)
    psi = TAU0 * source['covariance'] + within + shift
    nu = d + 1 + TAU0 + len(labels)
    df = nu - d + 1
    shape_factors = (1 + 1 / info) / df
    loglik = np.column_stack([multivariate_t.logpdf(query, loc=means[c],
        shape=psi * shape_factors[c], df=df) for c in range(classes)])
    assert np.isfinite(loglik).all()
    return dict(counts=counts, means=means, loglikelihood=loglik)


def run(lam, smoke=False):
    out = {'ports': {}}
    for port in sorted(p.name[:-4] for p in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{port}.npz', allow_pickle=False)
        if 'skipped' in d.files:
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
            zq = (feats[qpos] - source_model['center']) / source_model['scale']
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
                    # M2 冻结版 → M3 后验 → 伪均值
                    p1 = predict_with_pseudo(source_model, feats[ann], ay, feats[qpos],
                                             np.zeros((K, zq.shape[1])), np.zeros(K), 0.0)
                    f1 = lik.class_mass(p1['loglikelihood'], p1['counts'])
                    ratio0 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
                    posterior = f1['posterior']
                    pseudo_n = 0.1 * posterior.sum(axis=0)
                    pseudo_means = np.zeros((K, zq.shape[1]))
                    for c in range(K):
                        if posterior[:, c].sum() > 0:
                            pseudo_means[c] = (posterior[:, c, None] * zq).sum(0) / posterior[:, c].sum()
                    # 回灌版 M2 → M3
                    p2 = predict_with_pseudo(source_model, feats[ann], ay, feats[qpos],
                                             pseudo_means, pseudo_n, lam)
                    f2 = lik.class_mass(p2['loglikelihood'], p2['counts'])
                    s2 = lik.robust_scores(f2['ratio'], b, e, observed)
                    ratio2 = project_class_evidence_rt(p2['loglikelihood'], f2['prior'])
                    s3 = lik.robust_scores(ratio2, b, e, observed)
                    arms = {'flat': s_flat, 'covariance_likelihood': s2, 'class_projection': s3}
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
        per = rec.groupby(['arm', 'mode']).ba.agg('mean').reset_index()
        bf = per[(per['arm'] == 'flat') & (per['mode'] == 'matched')]['ba'].iloc[0]
        v = per[(per['arm'] == 'class_projection') & (per['mode'] == 'matched')]['ba'].iloc[0]
        nat = float(per[(per['arm'] == 'class_projection') & (per['mode'] == 'natural')]['ba'].iloc[0]
                    - per[(per['arm'] == 'flat') & (per['mode'] == 'natural')]['ba'].iloc[0])
        out['ports'][port] = {'n': int(d['n']), 'cp_flat': v - bf, 'natural': nat}
        print(f'lam={lam} {port}: {100*(v-bf):+.3f}pp', flush=True)
        if smoke:
            break
    plist = [p for p in out['ports'] if 'cp_flat' in out['ports'][p]]
    cp = np.array([out['ports'][p]['cp_flat'] for p in plist])
    nat = np.array([out['ports'][p]['natural'] for p in plist])
    res = {'lam': lam, 'n_ports': len(plist), 'cp_flat_mean': float(cp.mean()),
           'cp_flat_positive': int((cp > 0).sum()), 'cp_flat_worst': float(cp.min()),
           'natural_mean': float(nat.mean())}
    (OUT / f'summary_lam_{lam}.json').write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
    (OUT / f'result_lam_{lam}.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    run(a.lam, smoke=a.smoke)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
