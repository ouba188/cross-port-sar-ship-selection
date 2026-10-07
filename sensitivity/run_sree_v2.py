"""SREE v2：因子模型（类专属载荷 β_c）+ 修测量口径（按预测类是否观察拆分）。
门控 = sign(ℓ(b) - ℓ(e))。未观察类均值 = μ_c + β_c·f_T（f_T 由观察类估、β_c 源侧学）。
用法：python run_sree_v2.py [--smoke]
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


def fit_factor_model(z_src, y_src, p_src, base_means):
    """源侧学类专属载荷 β_c：δ_{c,p} = μ_cp − base_means ≈ β_c·f_p。base_means 须与 shared_cov 一致（sm['means']）。"""
    classes = np.unique(y_src); ports = np.unique(p_src)
    mu_cp = {}
    for c in classes:
        for p in ports:
            m = (y_src == c) & (p_src == p)
            if m.sum() >= 2:
                mu_cp[(c, p)] = z_src[m].mean(0)
    f_p = {}
    for p in ports:
        ds = [mu_cp[(c, p)] - base_means[c] for c in classes if (c, p) in mu_cp]
        if ds:
            f_p[p] = np.mean(ds, 0)
    beta = {}
    for c in classes:
        num = den = 0.0
        for p in ports:
            if (c, p) in mu_cp and p in f_p:
                fp = f_p[p]; d = mu_cp[(c, p)] - base_means[c]
                num += float(d @ fp); den += float(fp @ fp)
        beta[c] = num / (den + 1e-9)
    return beta


def likelihood_with_means(source, ann_x, labels, query_x, means_override=None):
    """shared_cov 的似然计算，means 可用 override 替换（对未观察类）。"""
    z = (np.asarray(ann_x) - source['center']) / source['scale']
    query = (np.asarray(query_x) - source['center']) / source['scale']
    d = source['means'].shape[1]
    counts = np.bincount(labels, minlength=K)
    sums = np.zeros((K, d)); np.add.at(sums, labels, z)
    information = TAU + counts
    means = (TAU * source['means'] + sums) / information[:, None]
    if means_override is not None:
        means = means.copy()
        means[~np.isnan(means_override)] = means_override[~np.isnan(means_override)]
    within, shift = np.zeros((d, d)), np.zeros((d, d))
    for c in np.flatnonzero(counts):
        avg = sums[c] / counts[c]; res = z[labels == c] - avg
        delta = avg - source['means'][c]
        within += res.T @ res
        shift += TAU * counts[c] / information[c] * np.outer(delta, delta)
    psi = TAU * source['covariance'] + within + shift
    nu = d + 1 + TAU + len(labels); df = nu - d + 1
    sf = (1 + 1 / information) / df
    return np.column_stack([multivariate_t.logpdf(query, loc=means[c], shape=psi * sf[c], df=df) for c in range(K)])


def gate(loglik, bq, eq):
    D = (bq != eq)
    dd = loglik[np.arange(len(bq)), eq] - loglik[np.arange(len(bq)), bq]
    return np.where(D, np.where(dd > 0, eq, bq), bq)


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
        z_src = (feats[source_pos] - sm['center']) / sm['scale']
        beta = fit_factor_model(z_src, y_ids[source_pos], ports_all[source_pos], sm['means'])

        row = {'port': p, 'ind_ba': [], 'sree_ba': [], 'ind_pu': [], 'sree_pu': [], 'ind_tu': [], 'sree_tu': []}
        for seed in SEEDS:
            ann = np.random.default_rng(seed).choice(qpos, ANNOT, replace=False)
            ay = y_ids[ann]
            in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
            ev = ~in_ann
            obs_cls = np.unique(ay)
            # 目标港口因子 f_T = 观察类平均漂移
            z_ann = (feats[ann] - sm['center']) / sm['scale']
            m_obs = np.array([z_ann[ay == c].mean(0) for c in obs_cls if (ay == c).sum() > 0])
            f_T = np.mean([m_obs[i] - sm['means'][c] for i, c in enumerate(obs_cls)], 0)
            # 未观察类均值 override = μ_c + β_c·f_T
            override = np.full((K, sm['means'].shape[1]), np.nan)
            for c in range(K):
                if c not in obs_cls:
                    override[c] = sm['means'][c] + beta.get(c, 1.0) * f_T
            ll_ind = shared_cov_predict(sm, feats[ann], ay, feats[qpos])['loglikelihood']
            ll_sree = likelihood_with_means(sm, feats[ann], ay, feats[qpos], means_override=override)
            g_ind = gate(ll_ind, bq, eq); g_sree = gate(ll_sree, bq, eq)
            row['ind_ba'].append(macro_ba(yq[ev], g_ind[ev])); row['sree_ba'].append(macro_ba(yq[ev], g_sree[ev]))
            # 预测类未观察（b 或 e 未在支持集）
            pu = (~np.isin(bq, obs_cls)) | (~np.isin(eq, obs_cls)); pu &= ev
            if pu.sum() > 0:
                row['ind_pu'].append((g_ind[pu] == yq[pu]).mean())
                row['sree_pu'].append((g_sree[pu] == yq[pu]).mean())
            # 真实类未观察
            tu = ~np.isin(yq, obs_cls) & ev
            if tu.sum() > 0:
                row['ind_tu'].append((g_ind[tu] == yq[tu]).mean())
                row['sree_tu'].append((g_sree[tu] == yq[tu]).mean())
        rows.append(row)
        print(f'{p}: ind={np.mean(row["ind_ba"]):.4f} sree={np.mean(row["sree_ba"]):.4f} '
              f'predUnobs ind={np.mean(row["ind_pu"]) if row["ind_pu"] else float("nan"):.3f}/sree={np.mean(row["sree_pu"]) if row["sree_pu"] else float("nan"):.3f}', flush=True)
        if smoke:
            break

    json.dump(rows, open('sree_v2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    ind = np.array([np.mean(r['ind_ba']) for r in rows]); sree = np.array([np.mean(r['sree_ba']) for r in rows])
    print(f'\n总体: 独立 BA={ind.mean():.4f}  共享 BA={sree.mean():.4f}')
    from scipy.stats import wilcoxon
    for key, name in [('pu', '预测类未观察'), ('tu', '真实类未观察')]:
        ii = np.array([np.mean(r[f'ind_{key}']) for r in rows if r[f'ind_{key}']])
        ss = np.array([np.mean(r[f'sree_{key}']) for r in rows if r[f'sree_{key}']])
        if len(ii) >= 2:
            dd = ss - ii
            w = wilcoxon(dd)
            print(f'{name}(港={len(ii)}): 独立={ii.mean():.4f} 共享={ss.mean():.4f} Δ={dd.mean():+.4f} 正港{int((dd>0).sum())}/{len(ii)} p={w.pvalue:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
