"""修正核验：精确覆盖 m 类 + M4 开/关诊断。
1. 精确控制覆盖：每港固定 top-m 类（query 类频排序），分层抽样保证【恰好 m 类观察】、每类 ≥1。
2. 4 臂（同一模型/评估集/seed）：
     A 独立+符号门控 (=M4关)    B 共享+符号门控 (=M4关)
     C 独立+s2 证据链 (=M4开)   D 共享+s2 证据链 (=M4开)
3. 报告 B−A（M4关）与 D−C（M4开）随 m 的响应曲线。
机制假设：若补全机制真，B−A 随 m 递减（低覆盖增益大）；D−C 平坦（M4 挡住未观察类专家选择）。
预注册判据：B−A 的 Spearman(m, B−A) < 0 且 m 小时 B−A 显著 > 0；D−C 全 m 平坦。
用法：python run_coverage_fix.py [--smoke]
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


def sign_gate(loglik, bq, eq, D):
    d = loglik[np.arange(len(bq)), eq] - loglik[np.arange(len(bq)), bq]
    return np.where(D & (d < 0), bq, eq)


def s2_gate(loglik, counts, bq, eq, D, observed):
    f1 = class_mass(loglik, counts)
    ratio2 = project_class_evidence_rt(loglik, f1['prior'])
    s2 = robust_scores(ratio2, bq, eq, observed)
    return np.where(D, np.where(s2 > 0, eq, bq), bq)


def stratified_annotate(rng, qpos, yq, topm, annot):
    """精确覆盖：topm 每类先取 1 个（保证恰好 m 类），其余从 topm 并集按自然频随机抽。"""
    parts = [rng.choice(qpos[yq == c], 1, replace=False) for c in topm]
    union = np.concatenate([qpos[yq == c] for c in topm])
    rest = rng.choice(union, annot - len(topm), replace=False)
    return np.concatenate(parts + [rest])


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

        freqs = np.bincount(yq, minlength=K)
        top_order = np.argsort(-freqs)

        row = {'port': p, 'true_ncls': len(np.unique(yq))}
        for m in MS:
            for arm in ['A', 'B', 'C', 'D']:
                row[f'{arm}_m{m}'] = []

        for seed in SEEDS:
            rng = np.random.default_rng(seed * 1000 + 13)
            for m in MS:
                topm = top_order[:m]
                if m > len(np.unique(yq)) or any(freqs[c] < 1 for c in topm):
                    for arm in 'ABCD':
                        row[f'{arm}_m{m}'].append(float('nan'))
                    continue
                ann = stratified_annotate(rng, qpos, yq, topm, ANNOT)
                ay = y_ids[ann]
                assert len(np.unique(ay)) == m, f'覆盖数 != m ({m})'
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

                # A/C（独立）
                row[f'A_m{m}'].append(macro_ba(yq[ev], sign_gate(loglik_ind, bq, eq, D)[ev]))
                row[f'C_m{m}'].append(macro_ba(yq[ev], s2_gate(loglik_ind, counts, bq, eq, D, observed)[ev]))
                # B/D（共享）
                f_T = target_factor_weighted(W, delta_obs, RANK, n_c)
                if f_T is not None:
                    mu = means_from_f(sm, W, f_T, obs_cls, obs_mean, n_c, KAPPA)
                    loglik_sh = loglik_full(query, psi, df, sf, mu)
                    row[f'B_m{m}'].append(macro_ba(yq[ev], sign_gate(loglik_sh, bq, eq, D)[ev]))
                    row[f'D_m{m}'].append(macro_ba(yq[ev], s2_gate(loglik_sh, counts, bq, eq, D, observed)[ev]))
                else:
                    row[f'B_m{m}'].append(float('nan')); row[f'D_m{m}'].append(float('nan'))

        rows.append(row)
        print(f'{p} (真类数={row["true_ncls"]}): ' + ' '.join(f'm{m}B-A={np.nanmean(np.array(row[f"B_m{m}"])-np.array(row[f"A_m{m}"])):+.4f}' for m in MS), flush=True)
        if smoke:
            break

    json.dump(rows, open('coverage_fix.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n=== B−A（M4关）与 D−C（M4开）随 m 响应 ===')
    curves = {}
    for tag, aa, bb in [('BA', 'A', 'B'), ('DC', 'C', 'D')]:
        print(f'--- {tag} ({aa}→{bb}) ---')
        c = {}
        for m in MS:
            a = np.array([np.nanmean(x[f'{aa}_m{m}']) for x in rows])
            b = np.array([np.nanmean(x[f'{bb}_m{m}']) for x in rows])
            diff = b - a
            diff = diff[~np.isnan(diff)]
            c[m] = diff
            if len(diff) >= 2:
                w = wilcoxon(diff)
                print(f'm={m}: Δ={diff.mean():+.4f} 正港{int((diff>0).sum())}/{len(diff)} p={w.pvalue:.4f}')
            else:
                print(f'm={m}: Δ={diff.mean():+.4f} (样本不足)')
        curves[tag] = c
        ms_arr = np.array(MS, dtype=float)
        means = np.array([c[m].mean() for m in MS])
        rho, pv = spearmanr(ms_arr, means)
        print(f'Spearman(m, {tag}) = {rho:.4f} (p={pv:.4f})')

    print('\n=== 预注册判据 ===')
    ba_means = np.array([curves['BA'][m].mean() for m in MS])
    rho_ba, pv_ba = spearmanr(np.array(MS, float), ba_means)
    dc_means = np.array([curves['DC'][m].mean() for m in MS])
    rho_dc, pv_dc = spearmanr(np.array(MS, float), dc_means)
    print(f'B−A: Spearman={rho_ba:.4f} p={pv_ba:.4f}  曲线: ' + ' '.join(f'{m}:{ba_means[i]:+.4f}' for i, m in enumerate(MS)))
    print(f'D−C: Spearman={rho_dc:.4f} p={pv_dc:.4f}  曲线: ' + ' '.join(f'{m}:{dc_means[i]:+.4f}' for i, m in enumerate(MS)))
    mech = rho_ba < -0.5 and (ba_means[0] > 0)
    blocked = abs(dc_means).max() < 0.003
    if mech and blocked:
        print('结论: ✅ 补全机制真实存在（B−A 随覆盖递减），但被 M4 挡住（D−C 平坦）→ 需放松 M4 再验')
    elif mech and not blocked:
        print('结论: ⚠️ 补全机制存在且 D−C 也有信号 → 共享适配有效')
    else:
        print('结论: ❌ 补全机制不成立（B−A 无覆盖依赖）→ 共享适配最终降级')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
