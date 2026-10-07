"""SREE 第一个实验：独立适配(当前 shared_cov) vs 共享适配(借用观察类漂移给未观察类)。
门控 = sign(ℓ(b) - ℓ(e))，按「观察类/未观察类」拆分评估。
用法：python run_sree_v1.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits
from scipy.stats import multivariate_t

from bdh_pixel_loss_controls.conditional_likelihood import fit_source, TAU
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import K

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
ANNOT = 64
SEEDS = [0, 1, 2, 3, 4]


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def wacc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def sree_likelihood(source, ann_x, labels, query_x, borrow=True):
    """shared_cov 的共享适配版：未观察类的均值从源先验 μ_c 换成 μ_c + δ̂_T（借用观察类平均漂移）。"""
    z = (np.asarray(ann_x) - source['center']) / source['scale']
    query = (np.asarray(query_x) - source['center']) / source['scale']
    d = source['means'].shape[1]
    counts = np.bincount(labels, minlength=K)
    sums = np.zeros((K, d))
    np.add.at(sums, labels, z)
    information = TAU + counts
    means = (TAU * source['means'] + sums) / information[:, None]
    if borrow:
        # 借用观察类平均漂移给未观察类
        obs = counts > 0
        if obs.any():
            delta_obs = means[obs] - source['means'][obs]  # 每观察类的实际漂移
            delta_T = delta_obs.mean(0)                     # 共享平均漂移（14 维）
            means[~obs] = source['means'][~obs] + delta_T
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
    loglik = np.column_stack([multivariate_t.logpdf(query, loc=means[c],
                                                    shape=psi * shape_factors[c], df=df) for c in range(K)])
    return loglik


def gate(loglik, bq, eq):
    D = (bq != eq)
    d = loglik[np.arange(len(bq)), eq] - loglik[np.arange(len(bq)), bq]
    return np.where(D, np.where(d > 0, eq, bq), bq)


def run(smoke=False):
    rows = []
    for p in sorted(x.name[:-4] for x in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
        qpos = d['qpos']; spos = d['spos']
        base = d['base_logits'].argmax(1); expert = d['expert_logits'].argmax(1)
        yq = y_ids[qpos]; bq = base[qpos]; eq = expert[qpos]
        query_n = len(qpos); source_n = min(query_n, len(spos))
        rng_s = np.random.default_rng(1000)
        strat = {c: rng_s.choice(spos[y_ids[spos] == c],
                                 max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False)
                 for c in np.unique(y_ids[spos])}
        source_pos = np.concatenate(list(strat.values()))
        sm = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])

        row = {'port': p, 'ind_ba': [], 'ind_acc': [], 'sree_ba': [], 'sree_acc': [],
               'ind_ba_unobs': [], 'sree_ba_unobs': []}
        for seed in SEEDS:
            ann = np.random.default_rng(seed).choice(qpos, ANNOT, replace=False)
            ay = y_ids[ann]
            in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
            ev = ~in_ann
            ll_ind = shared_cov_predict(sm, feats[ann], ay, feats[qpos])['loglikelihood']
            ll_sree = sree_likelihood(sm, feats[ann], ay, feats[qpos], borrow=True)
            g_ind = gate(ll_ind, bq, eq)
            g_sree = gate(ll_sree, bq, eq)
            row['ind_ba'].append(macro_ba(yq[ev], g_ind[ev]))
            row['ind_acc'].append(wacc(yq[ev], g_ind[ev]))
            row['sree_ba'].append(macro_ba(yq[ev], g_sree[ev]))
            row['sree_acc'].append(wacc(yq[ev], g_sree[ev]))
            # 未观察类拆分
            obs_cls = np.unique(ay)
            unobs = ~np.isin(yq, obs_cls) & ev
            if unobs.sum() > 0:
                row['ind_ba_unobs'].append((g_ind[unobs] == yq[unobs]).mean())
                row['sree_ba_unobs'].append((g_sree[unobs] == yq[unobs]).mean())
        rows.append(row)
        print(f'{p}: ind_BA={np.mean(row["ind_ba"]):.4f} sree_BA={np.mean(row["sree_ba"]):.4f} '
              f'ind_unobs={np.mean(row["ind_ba_unobs"]) if row["ind_ba_unobs"] else float("nan"):.3f} '
              f'sree_unobs={np.mean(row["sree_ba_unobs"]) if row["sree_ba_unobs"] else float("nan"):.3f}', flush=True)
        if smoke:
            break

    json.dump(rows, open('sree_v1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    ind = np.array([np.mean(r['ind_ba']) for r in rows]); sree = np.array([np.mean(r['sree_ba']) for r in rows])
    ind_a = np.array([np.mean(r['ind_acc']) for r in rows]); sree_a = np.array([np.mean(r['sree_acc']) for r in rows])
    print(f'\n独立 BA={ind.mean():.4f} ACC={ind_a.mean():.4f}')
    print(f'共享 BA={sree.mean():.4f} ACC={sree_a.mean():.4f}')
    # 未观察类
    ind_u = np.array([np.mean(r['ind_ba_unobs']) for r in rows if r['ind_ba_unobs']])
    sree_u = np.array([np.mean(r['sree_ba_unobs']) for r in rows if r['sree_ba_unobs']])
    print(f'未观察类(港数={len(ind_u)}): 独立 acc={ind_u.mean():.4f}  共享 acc={sree_u.mean():.4f}  Δ={sree_u.mean()-ind_u.mean():+.4f}')
    from scipy.stats import wilcoxon
    d_u = sree_u - ind_u
    w = wilcoxon(d_u) if len(d_u) >= 2 else None
    print(f'未观察类 paired: 正港 {int((d_u>0).sum())}/{len(d_u)}, Wilcoxon p={w.pvalue:.4f}' if w else 'n<2')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
