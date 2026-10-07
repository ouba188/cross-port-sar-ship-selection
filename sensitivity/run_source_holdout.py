"""源港留出：内层 LOPO 定秩 r 与收缩 κ（阈值已证无关，固定零阈值）。
对每个目标港 p：
  内层：对每个源港 p'（伪目标），从「其余 22 源港」拟合 W_c(r)，伪目标抽样 64 标签估 f_{p'}，
        共享适配 + 零阈值门控，测伪目标 eval 的 BA；按 (r,κ) 平均，选 argmax。
  外层：用 (r*,κ*) 对真目标 p 全 5 seed 评估。
对照：Fusion、s2、固定(r=2,κ=8)。
用法：python run_source_holdout.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits

from bdh_pixel_loss_controls.conditional_likelihood import fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import K, project_class_evidence_rt

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
ANNOT = 64
RANKS = [1, 2]
KAPPAS = [2.0, 4.0, 8.0, 16.0]
INNER_SEEDS = [0]
OUTER_SEEDS = [0, 1, 2, 3, 4]


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
    d = next(iter(delta_obs.values())).shape[0]
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


def loglik_eb(query, psi_inv, df, sf, means, bq, eq):
    d = query.shape[1]
    zQz = np.einsum('ij,jk,ik->i', query, psi_inv, query)
    a = means @ psi_inv
    b = np.einsum('ij,ij->i', means, a)
    ae, ab = a[eq], a[bq]
    be, bb = b[eq], b[bq]
    Me = zQz - 2 * np.einsum('ij,ij->i', ae, query) + be
    Mb = zQz - 2 * np.einsum('ij,ij->i', ab, query) + bb
    sf_e, sf_b = sf[eq], sf[bq]
    le = -0.5 * d * np.log(sf_e) - (df + d) / 2 * np.log1p(Me / (df * sf_e))
    lb = -0.5 * d * np.log(sf_b) - (df + d) / 2 * np.log1p(Mb / (df * sf_b))
    return le, lb


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


def delta_from(z, y, ports, sm_means):
    out = {}
    for c in np.unique(y):
        for pp in np.unique(ports):
            m = (y == c) & (ports == pp)
            if m.sum() >= 2:
                out[(c, pp)] = z[m].mean(0) - sm_means[c]
    return out


def shared_gate_ba(sm, W, f_T, obs_cls, obs_mean, n_c, psi_inv, df, sf, query, bq, eq, D, yq, ev, kappa):
    mu = means_from_f(sm, W, f_T, obs_cls, obs_mean, n_c, kappa)
    le, lb = loglik_eb(query, psi_inv, df, sf, mu, bq, eq)
    d = le - lb
    g = np.where(D & (d < 0), bq, eq)
    return macro_ba(yq[ev], g[ev])


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
        query_n = len(qpos)
        D = (bq != eq)

        # 外层源（全部 23 源港，分层抽样）
        source_n = min(query_n, len(spos))
        rng_s = np.random.default_rng(1000)
        strat = {c: rng_s.choice(spos[y_ids[spos] == c], max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False) for c in np.unique(y_ids[spos])}
        source_pos = np.concatenate(list(strat.values()))
        sm = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
        z_src = (feats[source_pos] - sm['center']) / sm['scale']

        src_ports = np.unique(ports_all[spos])

        # 内层留出：逐源港当伪目标
        inner_scores = {(r, k): [] for r in RANKS for k in KAPPAS}
        for pp in src_ports:
            if smoke and pp != src_ports[0]:
                continue
            ps_pos = spos[ports_all[spos] != pp]       # 伪源（22 港）
            pt_pos = spos[ports_all[spos] == pp]       # 伪目标样本
            if len(pt_pos) < 8:
                continue
            # 伪源 fit_source
            ps_n = min(len(pt_pos), len(ps_pos))
            ps_strat = {c: rng_s.choice(ps_pos[y_ids[ps_pos] == c], max(1, int(round(ps_n * (y_ids[ps_pos] == c).sum() / len(ps_pos)))), replace=False) for c in np.unique(y_ids[ps_pos])}
            ps_sel = np.concatenate(list(ps_strat.values()))
            smp = fit_source(feats[ps_sel], y_ids[ps_sel], ports_all[ps_sel])
            z_ps = (feats[ps_sel] - smp['center']) / smp['scale']
            delta_cp = delta_from(z_ps, y_ids[ps_sel], ports_all[ps_sel], smp['means'])
            if len(delta_cp) < 4:
                continue
            # 伪目标 query（pt_pos 全部样本）
            pt_feats = feats[pt_pos]; pt_y = y_ids[pt_pos]
            pt_bq = base[pt_pos]; pt_eq = expert[pt_pos]
            pt_D = (pt_bq != pt_eq)
            # 逐 seed
            for seed in INNER_SEEDS:
                ann = np.random.default_rng(seed).choice(len(pt_pos), min(ANNOT, len(pt_pos)), replace=False)
                ay = pt_y[ann]
                in_ann = np.zeros(len(pt_pos), bool); in_ann[ann] = True
                ev = ~in_ann
                obs_cls = set(np.unique(ay))
                n_c = np.bincount(ay, minlength=K)
                z_ann = (pt_feats[ann] - smp['center']) / smp['scale']
                obs_mean = {c: z_ann[ay == c].mean(0) for c in obs_cls}
                delta_obs = {c: obs_mean[c] - smp['means'][c] for c in obs_cls}
                r_ind = shared_cov_predict(smp, pt_feats[ann], ay, pt_feats)
                psi_inv = np.linalg.inv(r_ind['psi'])
                df = float(r_ind['df']); sf = r_ind['shape_factors']
                query_pt = (pt_feats - smp['center']) / smp['scale']
                for r in RANKS:
                    W = fit_factor_als(delta_cp, r)
                    f_T = target_factor_weighted(W, delta_obs, r, n_c)
                    if f_T is None:
                        continue
                    for k in KAPPAS:
                        ba = shared_gate_ba(smp, W, f_T, obs_cls, obs_mean, n_c, psi_inv, df, sf, query_pt, pt_bq, pt_eq, pt_D, pt_y, ev, k)
                        inner_scores[(r, k)].append(ba)

        # 选 (r*, k*)
        best = None; best_score = -np.inf
        for (r, k), vals in inner_scores.items():
            s = np.mean(vals) if vals else -np.inf
            if s > best_score:
                best_score = s; best = (r, k)
        r_star, k_star = best if best else (2, 8.0)

        # 外层最终评估（5 seed）
        delta_cp_full = delta_from(z_src, y_ids[source_pos], ports_all[source_pos], sm['means'])
        W_full = fit_factor_als(delta_cp_full, r_star)
        row = {'port': p, 'r_star': r_star, 'k_star': k_star, 'fus': [], 's2': [], 'shared': [], 'fixed28': []}
        for seed in OUTER_SEEDS:
            ann = np.random.default_rng(seed).choice(qpos, ANNOT, replace=False)
            ay = y_ids[ann]
            in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
            ev = ~in_ann
            # fusion
            row['fus'].append(macro_ba(yq[ev], eq[ev]))
            # s2
            s2 = full_s2(feats, y_ids, ports_all, qpos, spos, base, expert, ann)
            g_s2 = np.where(D, np.where(s2 > 0, eq, bq), bq)
            row['s2'].append(macro_ba(yq[ev], g_s2[ev]))
            # shared (r*, k*)
            obs_cls = set(np.unique(ay))
            n_c = np.bincount(ay, minlength=K)
            z_ann = (feats[ann] - sm['center']) / sm['scale']
            obs_mean = {c: z_ann[ay == c].mean(0) for c in obs_cls}
            delta_obs = {c: obs_mean[c] - sm['means'][c] for c in obs_cls}
            r_ind = shared_cov_predict(sm, feats[ann], ay, feats[qpos])
            psi_inv = np.linalg.inv(r_ind['psi']); df = float(r_ind['df']); sf = r_ind['shape_factors']
            query = (feats[qpos] - sm['center']) / sm['scale']
            f_T = target_factor_weighted(W_full, delta_obs, r_star, n_c)
            if f_T is not None:
                ba = shared_gate_ba(sm, W_full, f_T, obs_cls, obs_mean, n_c, psi_inv, df, sf, query, bq, eq, D, yq, ev, k_star)
                row['shared'].append(ba)
            else:
                row['shared'].append(float('nan'))
            # 固定 r=2 k=8
            W2 = fit_factor_als(delta_cp_full, 2)
            f_T2 = target_factor_weighted(W2, delta_obs, 2, n_c)
            if f_T2 is not None:
                ba2 = shared_gate_ba(sm, W2, f_T2, obs_cls, obs_mean, n_c, psi_inv, df, sf, query, bq, eq, D, yq, ev, 8.0)
                row['fixed28'].append(ba2)
            else:
                row['fixed28'].append(float('nan'))

        rows.append(row)
        print(f'{p}: r*={r_star} k*={k_star} fus={np.mean(row["fus"]):.4f} s2={np.mean(row["s2"]):.4f} '
              f'shared={np.nanmean(row["shared"]):.4f} fixed28={np.nanmean(row["fixed28"]):.4f}', flush=True)
        if smoke:
            break

    json.dump(rows, open('source_holdout.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    from scipy.stats import wilcoxon
    print('\n=== 汇总 ===')
    for k in ['fus', 's2', 'shared', 'fixed28']:
        v = np.array([np.nanmean(x[k]) for x in rows])
        print(f'{k}: BA={v.mean():.4f}')
    rs = np.array([x['r_star'] for x in rows]); ks = np.array([x['k_star'] for x in rows])
    print(f'r* 分布: {dict(zip(*np.unique(rs, return_counts=True)))}')
    print(f'k* 分布: {dict(zip(*np.unique(ks, return_counts=True)))}')
    def wil(a, b):
        va = np.array([np.nanmean(x[a]) for x in rows]); vb = np.array([np.nanmean(x[b]) for x in rows])
        dd = va - vb; m = ~np.isnan(dd)
        if m.sum() < 2:
            return
        w = wilcoxon(dd[m])
        print(f'{a} vs {b}: Δ={np.nanmean(dd):+.4f} 正港{int((dd>0).sum())}/{len(dd)} p={w.pvalue:.4f}')
    wil('shared', 'fus')
    wil('shared', 's2')
    wil('shared', 'fixed28')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
