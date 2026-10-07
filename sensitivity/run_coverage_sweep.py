"""受控覆盖扫描：固定观测类数 m（top-m 类），画 D−C 随 m 的响应曲线。
对每港：取 query 类频 top-m，抽样 64 标签，算 C(独立s2) vs D(共享s2)。
机制假设：共享适配填补 8−m 个未观察类均值 → m 越小，D−C 越大（正），m 越大趋 0/负。
预注册成功标准：D−C(m) 随 m 单调递减（Spearman<0），且 m=2 的 D−C 显著>0、m≥6 的 D−C≤0。
用法：python run_coverage_sweep.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits
from scipy.stats import multivariate_t, wilcoxon, spearmanr

from bdh_pixel_loss_controls.conditional_likelihood import fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import K, project_class_evidence_rt

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
ANNOT = 64
RANK = 2
KAPPA = 8.0
SEEDS = [0, 1, 2, 3, 4]
MS = [2, 3, 4, 5, 6, 7]


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def fit_factor_als(delta_cp, r, iters=50, restarts=3):
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


def target_factor_weighted(W, delta_obs, r, n_c):
    cs = [c for c in delta_obs if c in W and n_c[c] > 0 and np.linalg.norm(W[c]) > 1e-9]
    if not cs:
        return None
    A = sum(n_c[c] * W[c].T @ W[c] for c in cs)
    b = sum(n_c[c] * W[c].T @ delta_obs[c] for c in cs)
    return np.linalg.solve(A + 1e-6 * np.eye(r), b)


def means_from_f(sm, W, f, obs_cls, obs_mean, n_c, kappa):
    mu = np.zeros_like(sm['means'])
    for c in range(K):
        shared = sm['means'][c] + (W[c] @ f if c in W and np.linalg.norm(W[c]) > 1e-9 else 0.0)
        if c in obs_cls and n_c[c] > 0:
            mu[c] = (kappa * shared + n_c[c] * obs_mean[c]) / (kappa + n_c[c])
        else:
            mu[c] = shared
    return mu


def loglik_full(query, psi, df, sf, means):
    return np.column_stack([multivariate_t.logpdf(query, loc=means[c], shape=psi * sf[c], df=df) for c in range(K)])


def delta_from(z, y, ports, sm_means):
    out = {}
    for c in np.unique(y):
        for pp in np.unique(ports):
            m = (y == c) & (ports == pp)
            if m.sum() >= 2:
                out[(c, pp)] = z[m].mean(0) - sm_means[c]
    return out


def s2_gate(loglik, counts, bq, eq, D, observed):
    f1 = class_mass(loglik, counts)
    ratio2 = project_class_evidence_rt(loglik, f1['prior'])
    s2 = robust_scores(ratio2, bq, eq, observed)
    return np.where(D, np.where(s2 > 0, eq, bq), bq)


def run(smoke=False):
    ports = sorted(x.name[:-4] for x in CACHE.glob('*.npz'))
    ports = [p for p in ports if p != 'Melbourne']
    rows = []
    for p in ports:
        d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
        feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
        qpos = d['qpos']; spos = d['spos']
        base = d['base_logits'].argmax(1); expert = d['expert_logits'].argmax(1)
        yq = y_ids[qpos]; bq = base[qpos]; eq = expert[qpos]
        query_n = len(qpos); D = (bq != eq)

        source_n = min(query_n, len(spos))
        rng_s = np.random.default_rng(1000)
        strat = {c: rng_s.choice(spos[y_ids[spos] == c], max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False) for c in np.unique(y_ids[spos])}
        source_pos = np.concatenate(list(strat.values()))
        sm = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
        z_src = (feats[source_pos] - sm['center']) / sm['scale']
        delta_cp = delta_from(z_src, y_ids[source_pos], ports_all[source_pos], sm['means'])
        W = fit_factor_als(delta_cp, RANK)
        query = (feats[qpos] - sm['center']) / sm['scale']

        # 类频排序（top-m 用）
        freqs = np.bincount(yq, minlength=K)
        top_order = np.argsort(-freqs)

        row = {'port': p}
        for m in MS:
            row[f'm{m}'] = []

        for seed in SEEDS:
            rng = np.random.default_rng(seed * 1000 + 13)
            for m in MS:
                topm = top_order[:m]
                cand = qpos[np.isin(yq, topm)]
                if len(cand) < ANNOT:
                    row[f'm{m}'].append(float('nan'))
                    continue
                ann = rng.choice(cand, ANNOT, replace=False)
                ay = y_ids[ann]
                in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
                ev = ~in_ann
                obs_cls = set(np.unique(ay))
                n_c = np.bincount(ay, minlength=K)
                observed = n_c > 0
                z_ann = (feats[ann] - sm['center']) / sm['scale']
                obs_mean = {c: z_ann[ay == c].mean(0) for c in obs_cls}
                delta_obs = {c: obs_mean[c] - sm['means'][c] for c in obs_cls}
                r_ind = shared_cov_predict(sm, feats[ann], ay, feats[qpos])
                loglik_ind = r_ind['loglikelihood']
                counts = r_ind['counts']
                psi = r_ind['psi']; df = float(r_ind['df']); sf = r_ind['shape_factors']

                gC = s2_gate(loglik_ind, counts, bq, eq, D, observed)
                baC = macro_ba(yq[ev], gC[ev])

                f_T = target_factor_weighted(W, delta_obs, RANK, n_c)
                if f_T is not None:
                    mu = means_from_f(sm, W, f_T, obs_cls, obs_mean, n_c, KAPPA)
                    loglik_sh = loglik_full(query, psi, df, sf, mu)
                    gD = s2_gate(loglik_sh, counts, bq, eq, D, observed)
                    baD = macro_ba(yq[ev], gD[ev])
                    row[f'm{m}'].append(baD - baC)
                else:
                    row[f'm{m}'].append(float('nan'))

        rows.append(row)
        print(f'{p}: ' + ' '.join(f'm{m}={np.nanmean(row[f"m{m}"]):+.4f}' for m in MS), flush=True)
        if smoke:
            break

    json.dump(rows, open('coverage_sweep.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n=== D−C 随观测类数 m 的响应（逐港配对均值）===')
    curve = {}
    for m in MS:
        vals = np.array([np.nanmean(x[f'm{m}']) for x in rows])
        vals = vals[~np.isnan(vals)]
        curve[m] = vals
        if len(vals) >= 2:
            w = wilcoxon(vals)
            print(f'm={m}: D−C={vals.mean():+.4f} 正港{int((vals>0).sum())}/{len(vals)} p={w.pvalue:.4f}')
        else:
            print(f'm={m}: D−C={vals.mean():+.4f} (样本不足)')

    # 单调性：Spearman(m, D−C(m))
    ms_vals = np.array(MS, dtype=float)
    dc_means = np.array([curve[m].mean() for m in MS])
    rho, pv = spearmanr(ms_vals, dc_means)
    print(f'\nSpearman(m, D−C) = {rho:.4f} (p={pv:.4f})  → {"单调递减 ✅" if rho < 0 else "非单调"}')
    print('响应曲线:', ' '.join(f'{m}:{dc_means[i]:+.4f}' for i, m in enumerate(MS)))

    print('\n=== 预注册判据 ===')
    ok_mono = rho < 0
    ok_low = curve[2].mean() > 0 and wilcoxon(curve[2]).pvalue < 0.05 if len(curve[2]) >= 2 else False
    high_vals = np.concatenate([curve[m] for m in [6, 7]])
    ok_high = high_vals.mean() <= 0
    print(f'单调递减: {ok_mono} | m=2 显著>0: {ok_low} (D−C={curve[2].mean():+.4f}) | m≥6 ≤0: {ok_high} (D−C={high_vals.mean():+.4f})')
    print(f'结论: {"✅ 机制成立（覆盖越不全，共享适配增益越大）" if (ok_mono and ok_low and ok_high) else "❌ 机制不成立/证据不足"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
