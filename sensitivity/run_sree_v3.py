"""SREE v3：完整因子模型 W_c f_p（ALS 拟合，r∈{1,2}）+ 最小二乘 f_T。
基准统一 sm['means']。门控 = sign(ℓ(b)-ℓ(e))，按预测类是否观察拆分。
用法：python run_sree_v3.py [--smoke]
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
SEEDS = list(range(20))
RS = [1, 2]


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def wacc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def fit_factor_als(delta_cp, r, iters=50, restarts=3):
    """ALS 拟合 δ_{c,p} = W_c f_p。随机初始化 + 多重启，避免零初始化压塌高秩。"""
    classes = sorted({c for (c, p) in delta_cp})
    ports = sorted({p for (c, p) in delta_cp})
    d = next(iter(delta_cp.values())).shape[0]
    best = None; best_err = np.inf
    for rs in range(restarts):
        rng = np.random.default_rng(rs)
        f_p = {p: rng.normal(0, 0.1, r) for p in ports}
        W = {c: np.zeros((d, r)) for c in classes}
        for _ in range(iters):
            for c in classes:
                ps = [p for p in ports if (c, p) in delta_cp]
                if len(ps) < r + 1:
                    continue
                F = np.stack([f_p[p] for p in ps]); D = np.stack([delta_cp[(c, p)] for p in ps])
                W[c] = np.linalg.lstsq(F, D, rcond=None)[0].T
            for p in ports:
                cs = [c for c in classes if (c, p) in delta_cp and np.linalg.norm(W[c]) > 1e-9]
                if not cs:
                    continue
                A = sum(W[c].T @ W[c] for c in cs)
                b = sum(W[c].T @ delta_cp[(c, p)] for c in cs)
                f_p[p] = np.linalg.solve(A + 1e-6 * np.eye(r), b)
        err = sum(float(np.sum((delta_cp[(c, p)] - W[c] @ f_p[p]) ** 2)) for (c, p) in delta_cp)
        if err < best_err:
            best_err = err; best = W
    return best


def target_factor(W, delta_obs, r):
    cs = [c for c in delta_obs if c in W and np.linalg.norm(W[c]) > 1e-9]
    if len(cs) < 1:
        return None
    A = sum(W[c].T @ W[c] for c in cs)
    b = sum(W[c].T @ delta_obs[c] for c in cs)
    return np.linalg.solve(A + 1e-6 * np.eye(r), b)


def likelihood_with_means(source, ann_x, labels, query_x, means_override=None):
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
        # 源侧 δ_{c,p}
        delta_cp = {}
        for c in np.unique(y_ids[source_pos]):
            for pp in np.unique(ports_all[source_pos]):
                m = (y_ids[source_pos] == c) & (ports_all[source_pos] == pp)
                if m.sum() >= 2:
                    delta_cp[(c, pp)] = z_src[m].mean(0) - sm['means'][c]
        Ws = {r: fit_factor_als(delta_cp, r) for r in RS}

        row = {'port': p, 'ind_ba': [], 'ind_pu': [], 'ind_tu': []}
        for r in RS:
            row[f'sree{r}_ba'] = []; row[f'sree{r}_pu'] = []; row[f'sree{r}_tu'] = []
        for seed in SEEDS:
            ann = np.random.default_rng(seed).choice(qpos, ANNOT, replace=False)
            ay = y_ids[ann]
            in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
            ev = ~in_ann
            obs_cls = np.unique(ay)
            n_c = np.bincount(ay, minlength=K)
            z_ann = (feats[ann] - sm['center']) / sm['scale']
            delta_obs = {c: z_ann[ay == c].mean(0) - sm['means'][c] for c in obs_cls}
            ll_ind = shared_cov_predict(sm, feats[ann], ay, feats[qpos])['loglikelihood']
            g_ind = gate(ll_ind, bq, eq)
            row['ind_ba'].append(macro_ba(yq[ev], g_ind[ev]))
            pu = (~np.isin(bq, obs_cls)) | (~np.isin(eq, obs_cls)); pu &= ev
            tu = ~np.isin(yq, obs_cls) & ev
            if pu.sum() > 0:
                row['ind_pu'].append((g_ind[pu] == yq[pu]).mean())
            if tu.sum() > 0:
                row['ind_tu'].append((g_ind[tu] == yq[tu]).mean())
            for r in RS:
                override = np.full((K, sm['means'].shape[1]), np.nan)
                f_T = target_factor(Ws[r], delta_obs, r)
                if f_T is not None:
                    for c in range(K):
                        if c not in obs_cls and c in Ws[r] and np.linalg.norm(Ws[r][c]) > 1e-9:
                            override[c] = sm['means'][c] + Ws[r][c] @ f_T
                ll_s = likelihood_with_means(sm, feats[ann], ay, feats[qpos], means_override=override)
                g_s = gate(ll_s, bq, eq)
                row[f'sree{r}_ba'].append(macro_ba(yq[ev], g_s[ev]))
                if pu.sum() > 0:
                    row[f'sree{r}_pu'].append((g_s[pu] == yq[pu]).mean())
                if tu.sum() > 0:
                    row[f'sree{r}_tu'].append((g_s[tu] == yq[tu]).mean())
        rows.append(row)
        print(f'{p}: ind={np.mean(row["ind_ba"]):.4f} ' +
              ' '.join(f'sree{r}={np.mean(row[f"sree{r}_ba"]):.4f}' for r in RS), flush=True)
        if smoke:
            break

    json.dump(rows, open('sree_v3.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    from scipy.stats import wilcoxon
    ind = np.array([np.mean(x['ind_ba']) for x in rows])
    print(f'\n独立 BA={ind.mean():.4f}')
    for r in RS:
        s = np.array([np.mean(x[f'sree{r}_ba']) for x in rows])
        print(f'sree{r} BA={s.mean():.4f}  Δ={s.mean()-ind.mean():+.4f}')
    for r in RS:
        for key, name in [('pu', '预测类未观察'), ('tu', '真实类未观察')]:
            ii = np.array([np.mean(x[f'ind_{key}']) for x in rows if x[f'ind_{key}']])
            ss = np.array([np.mean(x[f'sree{r}_{key}']) for x in rows if x[f'sree{r}_{key}']])
            if len(ii) >= 2:
                dd = ss - ii
                w = wilcoxon(dd)
                print(f'sree{r} {name}(港={len(ii)}): 独立={ii.mean():.4f} 共享={ss.mean():.4f} Δ={dd.mean():+.4f} 正港{int((dd>0).sum())}/{len(ii)} p={w.pvalue:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
