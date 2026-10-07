"""SREE v5（正确版）：联合相对不确定性 + Fusion 默认门控。
对照用户 4 点受控修复：
1. 同一 support/test/seed 跑 Fusion / 完整 s2 / 独立似然门控 / 共享-联合门控。
2. 共享因子覆盖【所有类】，f_T 按 n_c 加权；观察类向共享预测收缩（κ）。
3. 从同一组共享参数采样 f_T → 联合后验 → d=ℓ(e)-ℓ(b)，z=d_mean/d_sd（Monte Carlo 自动含协方差→共同误差抵消）。
4. 默认 Fusion，z < -c 才切 VV；报挽回收益 + 错误切换损失。
用法：python run_sree_v5.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits

from bdh_pixel_loss_controls.conditional_likelihood import fit_source, TAU, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import K, project_class_evidence_rt

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
ANNOT = 64
SEEDS = [0, 1, 2, 3, 4]
RANK = 2
MC = 50          # Monte Carlo 样本数（f_T 后验）
KAPPA = 8.0      # 共享预测先验强度（源港留出确定 = 后续）
CS = [0.0, 0.5, 1.0, 1.5]


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
        return None, None
    A = sum(n_c[c] * W[c].T @ W[c] for c in cs)
    b = sum(n_c[c] * W[c].T @ delta_obs[c] for c in cs)
    f = np.linalg.solve(A + 1e-6 * np.eye(r), b)
    res = sum(n_c[c] * np.sum((delta_obs[c] - W[c] @ f) ** 2) for c in cs)
    dof = max(1, sum(n_c[c] for c in cs) - r)
    Sigma_f = (res / dof) * np.linalg.inv(A + 1e-6 * np.eye(r))
    return f, Sigma_f


def means_from_f(sm, W, f, obs_cls, obs_mean, n_c):
    """给定 f_T，返回所有类均值（观察类向共享预测收缩）。"""
    mu = np.zeros_like(sm['means'])
    for c in range(K):
        shared = sm['means'][c] + (W[c] @ f if c in W and np.linalg.norm(W[c]) > 1e-9 else 0.0)
        if c in obs_cls and n_c[c] > 0:
            mu[c] = (KAPPA * shared + n_c[c] * obs_mean[c]) / (KAPPA + n_c[c])
        else:
            mu[c] = shared
    return mu


def loglik_diff(query, psi_inv, df, sf, means, bq, eq):
    """ℓ(e)-ℓ(b)（不含常数项，差值中相消）。psi_inv 预计算。"""
    d = query.shape[1]
    zQz = np.einsum('ij,jk,ik->i', query, psi_inv, query)
    a = means @ psi_inv                    # K x d
    b = np.einsum('ij,ij->i', means, a)    # K
    ae, ab = a[eq], a[bq]
    be, bb = b[eq], b[bq]
    Me = zQz - 2 * np.einsum('ij,ij->i', ae, query) + be
    Mb = zQz - 2 * np.einsum('ij,ij->i', ab, query) + bb
    sf_e, sf_b = sf[eq], sf[bq]
    le = -0.5 * d * np.log(sf_e) - (df + d) / 2 * np.log1p(Me / (df * sf_e))
    lb = -0.5 * d * np.log(sf_b) - (df + d) / 2 * np.log1p(Mb / (df * sf_b))
    return le - lb


def full_s2(feats, y_ids, ports_all, qpos, spos, base, expert, ann):
    query_n = len(qpos); source_n = min(query_n, len(spos))
    rng_s = np.random.default_rng(1000)
    strat = {c: rng_s.choice(spos[y_ids[spos] == c], max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False) for c in np.unique(y_ids[spos])}
    source_pos = np.concatenate(list(strat.values()))
    ay = y_ids[ann]
    bq = base[qpos]; eq = expert[qpos]
    observed = np.bincount(ay, minlength=K) > 0
    sm = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
    p1 = shared_cov_predict(sm, feats[ann], ay, feats[qpos])
    f1 = class_mass(p1['loglikelihood'], p1['counts'])
    ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
    return robust_scores(ratio2, bq, eq, observed)


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
        strat = {c: rng_s.choice(spos[y_ids[spos] == c], max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False) for c in np.unique(y_ids[spos])}
        source_pos = np.concatenate(list(strat.values()))
        sm = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
        z_src = (feats[source_pos] - sm['center']) / sm['scale']
        delta_cp = {}
        for c in np.unique(y_ids[source_pos]):
            for pp in np.unique(ports_all[source_pos]):
                m = (y_ids[source_pos] == c) & (ports_all[source_pos] == pp)
                if m.sum() >= 2:
                    delta_cp[(c, pp)] = z_src[m].mean(0) - sm['means'][c]
        W = fit_factor_als(delta_cp, RANK)

        row = {'port': p, 'fus_ba': [], 'fus_acc': [], 's2_ba': [], 's2_acc': [],
               'ind_ba': [], 'ind_acc': []}
        for c in CS:
            row[f'jnt{c}_ba'] = []; row[f'jnt{c}_acc'] = []
            row[f'jnt{c}_rec'] = []; row[f'jnt{c}_wsw'] = []
            row[f'jnt{c}_nrec'] = []; row[f'jnt{c}_nwsw'] = []

        for seed in SEEDS:
            ann = np.random.default_rng(seed).choice(qpos, ANNOT, replace=False)
            ay = y_ids[ann]
            in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
            ev = ~in_ann
            obs_cls = set(np.unique(ay))
            n_c = np.bincount(ay, minlength=K)
            z_ann = (feats[ann] - sm['center']) / sm['scale']
            obs_mean = {c: z_ann[ay == c].mean(0) for c in obs_cls}
            delta_obs = {c: obs_mean[c] - sm['means'][c] for c in obs_cls}
            query = (feats[qpos] - sm['center']) / sm['scale']
            D = (bq != eq)

            # Fusion（默认专家，不切）
            g_fus = eq
            row['fus_ba'].append(macro_ba(yq[ev], g_fus[ev]))
            row['fus_acc'].append(wacc(yq[ev], g_fus[ev]))

            # 完整 s2
            s2 = full_s2(feats, y_ids, ports_all, qpos, spos, base, expert, ann)
            g_s2 = np.where(D, np.where(s2 > 0, eq, bq), bq)
            row['s2_ba'].append(macro_ba(yq[ev], g_s2[ev]))
            row['s2_acc'].append(wacc(yq[ev], g_s2[ev]))

            # 独立似然门控（sign(ℓ_e-ℓ_b)，shared_cov 均值）
            r_ind = shared_cov_predict(sm, feats[ann], ay, feats[qpos])
            psi_inv = np.linalg.inv(r_ind['psi'])
            d_ind = loglik_diff(query, psi_inv, float(r_ind['df']), r_ind['shape_factors'], r_ind['means'], bq, eq)
            g_ind = np.where(D, np.where(d_ind > 0, eq, bq), bq)
            row['ind_ba'].append(macro_ba(yq[ev], g_ind[ev]))
            row['ind_acc'].append(wacc(yq[ev], g_ind[ev]))

            # 共享-联合门控（Monte Carlo）
            f_T, Sigma_f = target_factor_weighted(W, delta_obs, RANK, n_c)
            if f_T is not None:
                rng_mc = np.random.default_rng(seed * 1000 + 7)
                f_samples = rng_mc.multivariate_normal(f_T, Sigma_f, size=MC)
                ds = []
                for m in range(MC):
                    mu = means_from_f(sm, W, f_samples[m], obs_cls, obs_mean, n_c)
                    ds.append(loglik_diff(query, psi_inv, float(r_ind['df']), r_ind['shape_factors'], mu, bq, eq))
                ds = np.array(ds)
                d_mean = ds.mean(0); d_sd = ds.std(0)
                for c in CS:
                    z = np.where(d_sd > 0, d_mean / np.maximum(d_sd, 1e-9), d_mean)
                    switch = D & (z < -c)
                    g = np.where(switch, bq, eq)
                    row[f'jnt{c}_ba'].append(macro_ba(yq[ev], g[ev]))
                    row[f'jnt{c}_acc'].append(wacc(yq[ev], g[ev]))
                    # 挽回/错误切换（仅 eval 分歧样本）
                    rec = D & ev & (bq == yq) & (eq != yq)
                    wsw = D & ev & (eq == yq) & (bq != yq)
                    row[f'jnt{c}_rec'].append(int((switch[ev] & rec[ev]).sum()))
                    row[f'jnt{c}_wsw'].append(int((switch[ev] & wsw[ev]).sum()))
                    row[f'jnt{c}_nrec'].append(int(rec[ev].sum()))
                    row[f'jnt{c}_nwsw'].append(int(wsw[ev].sum()))
            else:
                for c in CS:
                    row[f'jnt{c}_ba'].append(float('nan')); row[f'jnt{c}_acc'].append(float('nan'))
                    row[f'jnt{c}_rec'].append(0); row[f'jnt{c}_wsw'].append(0)
                    row[f'jnt{c}_nrec'].append(0); row[f'jnt{c}_nwsw'].append(0)

        rows.append(row)
        print(f'{p}: fus={np.mean(row["fus_ba"]):.4f} s2={np.mean(row["s2_ba"]):.4f} '
              f'ind={np.mean(row["ind_ba"]):.4f} ' + ' '.join(f'jnt{c}={np.nanmean(row[f"jnt{c}_ba"]):.4f}' for c in CS), flush=True)
        if smoke:
            break

    json.dump(rows, open('sree_v5.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n=== BA 汇总 ===')
    for k in ['fus_ba', 's2_ba', 'ind_ba'] + [f'jnt{c}_ba' for c in CS]:
        v = np.array([np.nanmean(x[k]) for x in rows])
        print(f'{k}: BA={v.mean():.4f}')
    from scipy.stats import wilcoxon
    s2 = np.array([np.mean(x['s2_ba']) for x in rows])
    for c in CS:
        j = np.array([np.nanmean(x[f'jnt{c}_ba']) for x in rows])
        dd = j - s2
        w = wilcoxon(dd[~np.isnan(dd)])
        print(f'jnt{c} vs s2: Δ={np.nanmean(dd):+.4f} 正港{int((dd>0).sum())}/{len(dd)} p={w.pvalue:.4f}')
    # 挽回/错误切换净收益（相对 Fusion）
    print('\n=== 挽回收益 / 错误切换（相对 Fusion，eval 分歧样本）===')
    for c in CS:
        rec = np.array([sum(x[f'jnt{c}_rec']) for x in rows])
        wsw = np.array([sum(x[f'jnt{c}_wsw']) for x in rows])
        nrec = np.array([sum(x[f'jnt{c}_nrec']) for x in rows])
        nwsw = np.array([sum(x[f'jnt{c}_nwsw']) for x in rows])
        rr = rec.sum() / max(1, nrec.sum()); wsr = wsw.sum() / max(1, nwsw.sum())
        print(f'jnt{c}: 挽回率={rr:.3f} 错误切换率={wsr:.3f} 净={rec.sum()-wsw.sum()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
