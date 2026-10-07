"""机制归因实验：证明/排除「联合误差建模有独立贡献」。
6 臂（同一 support/test/seed）：
  fusion          统一参照（默认专家）
  s2              统一参照（当前方法）
  shared0         共享参数【点估计】+ 零阈值   → 共享适配本身贡献
  jmean0          联合采样【均值】+ 零阈值      → 对参数不确定性取平均是否额外有用
  jconf_c         联合采样 + 置信门控(z<-c)     → 不确定性是否改善切换
  shuff_c         打乱采样配对 + 置信门控       → 共同误差协方差是否真有价值
关键判据：同挽回下是否减误切；shuff vs jconf 决定协方差价值。
用法：python run_mechanism_attribution.py [--smoke]
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
SEEDS = [0, 1, 2, 3, 4]
RANK = 2
MC = 50
KAPPA = 8.0
CS = [0.0, 0.5, 1.0]


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
    """加权 LS f_T。dof 修正：#观测类×特征维 − r（不是 Σn_c − r）。"""
    cs = [c for c in delta_obs if c in W and n_c[c] > 0 and np.linalg.norm(W[c]) > 1e-9]
    if not cs:
        return None, None
    d = next(iter(delta_obs.values())).shape[0]
    A = sum(n_c[c] * W[c].T @ W[c] for c in cs)
    b = sum(n_c[c] * W[c].T @ delta_obs[c] for c in cs)
    f = np.linalg.solve(A + 1e-6 * np.eye(r), b)
    res = sum(n_c[c] * np.sum((delta_obs[c] - W[c] @ f) ** 2) for c in cs)
    dof = max(1, len(cs) * d - r)
    Sigma_f = (res / dof) * np.linalg.inv(A + 1e-6 * np.eye(r))
    return f, Sigma_f


def means_from_f(sm, W, f, obs_cls, obs_mean, n_c):
    mu = np.zeros_like(sm['means'])
    for c in range(K):
        shared = sm['means'][c] + (W[c] @ f if c in W and np.linalg.norm(W[c]) > 1e-9 else 0.0)
        if c in obs_cls and n_c[c] > 0:
            mu[c] = (KAPPA * shared + n_c[c] * obs_mean[c]) / (KAPPA + n_c[c])
        else:
            mu[c] = shared
    return mu


def loglik_eb(query, psi_inv, df, sf, means, bq, eq):
    """返回 (ℓ_e, ℓ_b)（含常数项部分在差值中相消，仅需差则用 le-lb）。"""
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


def stat(g, yq, ev, D, bq, eq):
    """返回 (BA, ACC, 挽回数, 误切数, 可挽回数, 可误切数)。"""
    ba = macro_ba(yq[ev], g[ev])
    acc = wacc(yq[ev], g[ev])
    rec = D & ev & (bq == yq) & (eq != yq)
    wsw = D & ev & (eq == yq) & (bq != yq)
    switch = (g == bq) & D & ev          # 切到 base(VV) 的样本
    return ba, acc, int((switch & rec).sum()), int((switch & wsw).sum()), int(rec.sum()), int(wsw.sum())


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

        row = {'port': p}
        for arm in ['fusion', 's2', 'shared0', 'jmean0'] + [f'jconf{c}' for c in CS[1:]] + [f'shuff{c}' for c in CS]:
            row[arm] = {'ba': [], 'acc': [], 'rec': [], 'wsw': [], 'nrec': [], 'nwsw': []}

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

            # fusion
            ba, acc, rec, wsw, nrec, nwsw = stat(eq, yq, ev, D, bq, eq)
            row['fusion']['ba'].append(ba); row['fusion']['acc'].append(acc)
            # s2
            s2 = full_s2(feats, y_ids, ports_all, qpos, spos, base, expert, ann)
            g_s2 = np.where(D, np.where(s2 > 0, eq, bq), bq)
            ba, acc, rec, wsw, nrec, nwsw = stat(g_s2, yq, ev, D, bq, eq)
            row['s2']['ba'].append(ba); row['s2']['acc'].append(acc)
            row['s2']['rec'].append(rec); row['s2']['wsw'].append(wsw)
            row['s2']['nrec'].append(nrec); row['s2']['nwsw'].append(nwsw)

            # shared 点估计 + 零阈值
            r_ind = shared_cov_predict(sm, feats[ann], ay, feats[qpos])
            psi_inv = np.linalg.inv(r_ind['psi'])
            df = float(r_ind['df']); sf = r_ind['shape_factors']
            f_T, Sigma_f = target_factor_weighted(W, delta_obs, RANK, n_c)
            if f_T is not None:
                mu_pt = means_from_f(sm, W, f_T, obs_cls, obs_mean, n_c)
                le_pt, lb_pt = loglik_eb(query, psi_inv, df, sf, mu_pt, bq, eq)
                d_pt = le_pt - lb_pt
                g_shared0 = np.where(D & (d_pt < 0), bq, eq)
                ba, acc, rec, wsw, nrec, nwsw = stat(g_shared0, yq, ev, D, bq, eq)
                row['shared0']['ba'].append(ba); row['shared0']['acc'].append(acc)
                row['shared0']['rec'].append(rec); row['shared0']['wsw'].append(wsw)
                row['shared0']['nrec'].append(nrec); row['shared0']['nwsw'].append(nwsw)

                # Monte Carlo 采样 f_T
                rng_mc = np.random.default_rng(seed * 1000 + 7)
                f_samples = rng_mc.multivariate_normal(f_T, Sigma_f, size=MC)
                LE = np.zeros((MC, query_n)); LB = np.zeros((MC, query_n))
                for m in range(MC):
                    mu = means_from_f(sm, W, f_samples[m], obs_cls, obs_mean, n_c)
                    le, lb = loglik_eb(query, psi_inv, df, sf, mu, bq, eq)
                    LE[m] = le; LB[m] = lb
                d4 = LE - LB
                d4_mean = d4.mean(0); d4_sd = d4.std(0)
                # jmean0 = d_mean 零阈值
                g_jmean0 = np.where(D & (d4_mean < 0), bq, eq)
                ba, acc, rec, wsw, nrec, nwsw = stat(g_jmean0, yq, ev, D, bq, eq)
                row['jmean0']['ba'].append(ba); row['jmean0']['acc'].append(acc)
                row['jmean0']['rec'].append(rec); row['jmean0']['wsw'].append(wsw)
                row['jmean0']['nrec'].append(nrec); row['jmean0']['nwsw'].append(nwsw)
                # 打乱配对（全局置换 LB 采样序）
                perm = rng_mc.permutation(MC)
                d5 = LE - LB[perm]
                d5_mean = d5.mean(0); d5_sd = d5.std(0)
                for c in CS[1:]:
                    z4 = np.where(d4_sd > 0, d4_mean / np.maximum(d4_sd, 1e-9), d4_mean)
                    g = np.where(D & (z4 < -c), bq, eq)
                    ba, acc, rec, wsw, nrec, nwsw = stat(g, yq, ev, D, bq, eq)
                    row[f'jconf{c}']['ba'].append(ba); row[f'jconf{c}']['acc'].append(acc)
                    row[f'jconf{c}']['rec'].append(rec); row[f'jconf{c}']['wsw'].append(wsw)
                    row[f'jconf{c}']['nrec'].append(nrec); row[f'jconf{c}']['nwsw'].append(nwsw)
                for c in CS:
                    z5 = np.where(d5_sd > 0, d5_mean / np.maximum(d5_sd, 1e-9), d5_mean)
                    g = np.where(D & (z5 < -c), bq, eq)
                    ba, acc, rec, wsw, nrec, nwsw = stat(g, yq, ev, D, bq, eq)
                    row[f'shuff{c}']['ba'].append(ba); row[f'shuff{c}']['acc'].append(acc)
                    row[f'shuff{c}']['rec'].append(rec); row[f'shuff{c}']['wsw'].append(wsw)
                    row[f'shuff{c}']['nrec'].append(nrec); row[f'shuff{c}']['nwsw'].append(nwsw)
            else:
                for arm in ['shared0', 'jmean0'] + [f'jconf{c}' for c in CS[1:]] + [f'shuff{c}' for c in CS]:
                    for k in ['ba', 'acc', 'rec', 'wsw', 'nrec', 'nwsw']:
                        row[arm][k].append(float('nan') if k in ('ba', 'acc') else 0)

        rows.append(row)
        print(f'{p}: fus={np.mean(row["fusion"]["ba"]):.4f} s2={np.mean(row["s2"]["ba"]):.4f} '
              f'sh0={np.mean(row["shared0"]["ba"]):.4f} jm0={np.mean(row["jmean0"]["ba"]):.4f} '
              + ' '.join(f'jc{c}={np.nanmean(row[f"jconf{c}"]["ba"]):.4f}' for c in CS[1:])
              + ' ' + ' '.join(f'sh{c}={np.nanmean(row[f"shuff{c}"]["ba"]):.4f}' for c in CS), flush=True)
        if smoke:
            break

    json.dump(rows, open('mechanism_attribution.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    from scipy.stats import wilcoxon
    print('\n=== BA 汇总 ===')
    for arm in ['fusion', 's2', 'shared0', 'jmean0'] + [f'jconf{c}' for c in CS[1:]] + [f'shuff{c}' for c in CS]:
        v = np.array([np.nanmean(x[arm]['ba']) for x in rows])
        print(f'{arm}: BA={v.mean():.4f}')

    def wil(a, b):
        va = np.array([np.nanmean(x[a]['ba']) for x in rows])
        vb = np.array([np.nanmean(x[b]['ba']) for x in rows])
        dd = va - vb
        m = ~np.isnan(dd)
        if m.sum() < 2:
            print(f'{a} vs {b}: 样本不足（{m.sum()}港），跳过')
            return
        w = wilcoxon(dd[m])
        print(f'{a} vs {b}: Δ={np.nanmean(dd):+.4f} 正港{int((dd>0).sum())}/{len(dd)} p={w.pvalue:.4f}')

    print('\n=== 关键归因检验 ===')
    wil('shared0', 'fusion')      # 共享适配贡献
    wil('jmean0', 'shared0')      # 参数平均贡献
    for c in CS[1:]:
        wil(f'jconf{c}', 'jmean0')   # 置信门控贡献
    for c in CS:
        wil(f'shuff{c}', f'jconf{c}' if c > 0 else 'jmean0')  # 协方差价值

    print('\n=== 挽回/误切（相对 Fusion）===')
    for arm in ['s2', 'shared0', 'jmean0'] + [f'jconf{c}' for c in CS[1:]] + [f'shuff{c}' for c in CS]:
        rec = np.array([sum(x[arm]['rec']) for x in rows]).sum()
        wsw = np.array([sum(x[arm]['wsw']) for x in rows]).sum()
        nrec = np.array([sum(x[arm]['nrec']) for x in rows]).sum()
        nwsw = np.array([sum(x[arm]['nwsw']) for x in rows]).sum()
        print(f'{arm}: 挽回率={rec/max(1,nrec):.3f} 误切率={wsw/max(1,nwsw):.3f} 净={rec-wsw}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
