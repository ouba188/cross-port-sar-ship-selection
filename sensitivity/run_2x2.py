"""2×2 对照：均值适配（独立/共享）× 门控（符号/s2 证据链）。
  A: 独立均值 + 符号门控
  B: 共享均值 + 符号门控      （B−A = 共享结构本身贡献）
  C: 独立均值 + s2 证据链     （= 原始 s2）
  D: 共享均值 + s2 证据链     （唯一新增候选，D−C = 对最强方法的增量）
D 只替换均值，保持协方差(psi/df/sf)与决策规则一致，重算 EM/OT 证据。
预注册成功标准（D−C）：Wilcoxon p<0.05 且 ≥16/23 港为正 且 均值 Δ≥0.003。
按标注集类别覆盖（观测类数）分组报告。
用法：python run_2x2.py [--smoke]
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

from bdh_pixel_loss_controls.conditional_likelihood import fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import K, project_class_evidence_rt

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
ANNOT = 64
RANK = 2
KAPPA = 8.0
SEEDS = [0, 1, 2, 3, 4]


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def wacc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


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
    """完整 K 类 Student-t 对数似然（给定均值，协方差/df/sf 不变）。"""
    return np.column_stack([multivariate_t.logpdf(query, loc=means[c], shape=psi * sf[c], df=df) for c in range(K)])


def delta_from(z, y, ports, sm_means):
    out = {}
    for c in np.unique(y):
        for pp in np.unique(ports):
            m = (y == c) & (ports == pp)
            if m.sum() >= 2:
                out[(c, pp)] = z[m].mean(0) - sm_means[c]
    return out


def sign_gate(loglik, bq, eq, D):
    d = loglik[np.arange(len(bq)), eq] - loglik[np.arange(len(bq)), bq]
    return np.where(D & (d < 0), bq, eq)


def s2_gate(loglik, counts, bq, eq, D, observed):
    f1 = class_mass(loglik, counts)
    ratio2 = project_class_evidence_rt(loglik, f1['prior'])
    s2 = robust_scores(ratio2, bq, eq, observed)
    return np.where(D, np.where(s2 > 0, eq, bq), bq)


def rec_wsw(g, yq, ev, D, bq, eq):
    rec = D & ev & (bq == yq) & (eq != yq)
    wsw = D & ev & (eq == yq) & (bq != yq)
    switch = (g == bq) & D & ev
    return int((switch & rec).sum()), int((switch & wsw).sum()), int(rec.sum()), int(wsw.sum())


def boot_ci(diffs, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


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

        row = {'port': p, 'n_obs': [], 'A': [], 'B': [], 'C': [], 'D': [],
               'C_rec': [], 'C_wsw': [], 'D_rec': [], 'D_wsw': []}
        for seed in SEEDS:
            ann = np.random.default_rng(seed).choice(qpos, ANNOT, replace=False)
            ay = y_ids[ann]
            in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
            ev = ~in_ann
            obs_cls = set(np.unique(ay))
            n_c = np.bincount(ay, minlength=K)
            observed = n_c > 0
            row['n_obs'].append(len(obs_cls))
            z_ann = (feats[ann] - sm['center']) / sm['scale']
            obs_mean = {c: z_ann[ay == c].mean(0) for c in obs_cls}
            delta_obs = {c: obs_mean[c] - sm['means'][c] for c in obs_cls}
            r_ind = shared_cov_predict(sm, feats[ann], ay, feats[qpos])
            loglik_ind = r_ind['loglikelihood']
            counts = r_ind['counts']
            psi = r_ind['psi']; df = float(r_ind['df']); sf = r_ind['shape_factors']
            query = (feats[qpos] - sm['center']) / sm['scale']

            # A: 独立 + 符号
            gA = sign_gate(loglik_ind, bq, eq, D)
            row['A'].append(macro_ba(yq[ev], gA[ev]))
            # C: 独立 + s2
            gC = s2_gate(loglik_ind, counts, bq, eq, D, observed)
            row['C'].append(macro_ba(yq[ev], gC[ev]))
            cr, cw, _, _ = rec_wsw(gC, yq, ev, D, bq, eq)
            row['C_rec'].append(cr); row['C_wsw'].append(cw)

            # B/D: 共享均值
            f_T = target_factor_weighted(W, delta_obs, RANK, n_c)
            if f_T is not None:
                mu = means_from_f(sm, W, f_T, obs_cls, obs_mean, n_c, KAPPA)
                loglik_sh = loglik_full(query, psi, df, sf, mu)
                gB = sign_gate(loglik_sh, bq, eq, D)
                row['B'].append(macro_ba(yq[ev], gB[ev]))
                gD = s2_gate(loglik_sh, counts, bq, eq, D, observed)
                row['D'].append(macro_ba(yq[ev], gD[ev]))
                dr, dw, _, _ = rec_wsw(gD, yq, ev, D, bq, eq)
                row['D_rec'].append(dr); row['D_wsw'].append(dw)
            else:
                row['B'].append(float('nan')); row['D'].append(float('nan'))
                row['D_rec'].append(0); row['D_wsw'].append(0)

        rows.append(row)
        print(f'{p}: n_obs={np.mean(row["n_obs"]):.1f} A={np.mean(row["A"]):.4f} B={np.nanmean(row["B"]):.4f} '
              f'C={np.mean(row["C"]):.4f} D={np.nanmean(row["D"]):.4f}  (B-A={np.nanmean(row["B"])-np.mean(row["A"]):+.4f} D-C={np.nanmean(row["D"])-np.mean(row["C"]):+.4f})', flush=True)
        if smoke:
            break

    json.dump(rows, open('run_2x2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    from scipy.stats import wilcoxon

    def arr(k):
        return np.array([np.nanmean(x[k]) for x in rows])

    print('\n=== BA 汇总 ===')
    for k in ['A', 'B', 'C', 'D']:
        print(f'{k}: {arr(k).mean():.4f}')

    print('\n=== 关键比较 ===')
    # B - A（共享结构本身贡献）
    dBA = arr('B') - arr('A')
    m = ~np.isnan(dBA)
    w = wilcoxon(dBA[m])
    ci = boot_ci(dBA[m])
    print(f'B−A: Δ={dBA[m].mean():+.4f} 正港{int((dBA>0).sum())}/{len(dBA)} p={w.pvalue:.4f} CI={ci[0]:+.4f}~{ci[1]:+.4f}')
    # D - C（对 s2 的增量）
    dDC = arr('D') - arr('C')
    m = ~np.isnan(dDC)
    w = wilcoxon(dDC[m])
    ci = boot_ci(dDC[m])
    print(f'D−C: Δ={dDC[m].mean():+.4f} 正港{int((dDC>0).sum())}/{len(dDC)} p={w.pvalue:.4f} CI={ci[0]:+.4f}~{ci[1]:+.4f}')

    print('\n=== 挽回/误切（相对 Fusion，C vs D）===')
    for k in ['C', 'D']:
        rec = sum(sum(x[f'{k}_rec']) for x in rows); wsw = sum(sum(x[f'{k}_wsw']) for x in rows)
        print(f'{k}: 挽回={rec} 误切={wsw} 净={rec-wsw}')

    print('\n=== 按类别覆盖分组（观测类数）===')
    nobs = np.array([np.mean(x['n_obs']) for x in rows])
    groups = [('低覆盖 2-3类', nobs <= 3.5), ('中覆盖 4-5类', (nobs > 3.5) & (nobs <= 5.5)), ('高覆盖 6-8类', nobs > 5.5)]
    for name, mask in groups:
        if mask.sum() == 0:
            continue
        dDC_g = dDC[mask]; m = ~np.isnan(dDC_g)
        if m.sum() < 2:
            print(f'{name}({mask.sum()}港): D−C 样本不足')
            continue
        print(f'{name}({mask.sum()}港): D−C Δ={dDC_g[m].mean():+.4f} 正港{int((dDC_g>0).sum())}/{len(dDC_g)}')

    print('\n=== 预注册判据（D−C）===')
    m = ~np.isnan(dDC)
    dd = dDC[m]
    ok_p = wilcoxon(dd).pvalue < 0.05
    ok_sign = int((dd > 0).sum()) >= 16
    ok_mag = dd.mean() >= 0.003
    print(f'p<0.05: {ok_p} | ≥16港为正: {ok_sign} | Δ≥0.003: {ok_mag} (Δ={dd.mean():+.4f})')
    print(f'判据结论: {"保留（共享适配是有效升级）" if (ok_p and ok_sign and ok_mag) else "降级（共享适配为消融，不构成核心升级）"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
