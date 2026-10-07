"""SREE v4：联合相对不确定性（置信边际门控）。
d̂=ℓ(b)−ℓ(e)；σ_d² = Mahal_b/(τ+n_b) + Mahal_e/(τ+n_e)（后验均值不确定度 × 马氏距离）。
门控：d̂/σ_d > c。c=0=原始符号门控。未观察类 n=0 → σ_d 大 → 保守不切。
扫 c ∈ {0, 0.5, 1.0, 1.5}。判据：c>0 在预测类未观察上须优于 c=0。
用法：python run_sree_v4.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits

from bdh_pixel_loss_controls.conditional_likelihood import fit_source, TAU
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import K

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
ANNOT = 64
SEEDS = [0, 1, 2, 3, 4]
CS = [0.0, 0.5, 1.0, 1.5]


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def wacc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


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

        row = {'port': p}
        for c in CS:
            row[f'c{c}_ba'] = []; row[f'c{c}_pu'] = []
        for seed in SEEDS:
            ann = np.random.default_rng(seed).choice(qpos, ANNOT, replace=False)
            ay = y_ids[ann]
            in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
            ev = ~in_ann
            obs_cls = np.unique(ay)
            n_c = np.bincount(ay, minlength=K)
            r = shared_cov_predict(sm, feats[ann], ay, feats[qpos])
            loglik = r['loglikelihood']; means = r['means']; psi = r['psi']
            query = (feats[qpos] - sm['center']) / sm['scale']
            # 马氏距离（用共享 psi）
            psi_inv = np.linalg.inv(psi)
            Mahal_b = np.einsum('ij,jk,ik->i', query - means[bq], psi_inv, query - means[bq])
            Mahal_e = np.einsum('ij,jk,ik->i', query - means[eq], psi_inv, query - means[eq])
            dhat = loglik[np.arange(query_n), eq] - loglik[np.arange(query_n), bq]
            sigma = np.sqrt(Mahal_b / (TAU + n_c[bq]) + Mahal_e / (TAU + n_c[eq]))
            D = (bq != eq)
            pu = (~np.isin(bq, obs_cls)) | (~np.isin(eq, obs_cls)); pu &= ev
            for c in CS:
                z = np.where(sigma > 0, dhat / np.maximum(sigma, 1e-9), dhat)
                switch = D & (z > c)
                pred = np.where(switch, eq, bq)
                row[f'c{c}_ba'].append(macro_ba(yq[ev], pred[ev]))
                if pu.sum() > 0:
                    row[f'c{c}_pu'].append((pred[pu] == yq[pu]).mean())
        rows.append(row)
        print(f'{p}: ' + ' '.join(f'c{c}={np.mean(row[f"c{c}_ba"]):.4f}' for c in CS), flush=True)
        if smoke:
            break

    json.dump(rows, open('sree_v4.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for c in CS:
        ba = np.array([np.mean(x[f'c{c}_ba']) for x in rows])
        pu = np.array([np.mean(x[f'c{c}_pu']) for x in rows if x[f'c{c}_pu']])
        print(f'c={c}: BA={ba.mean():.4f}  预测类未观察 acc={pu.mean():.4f} (港{len(pu)})')
    from scipy.stats import wilcoxon
    for c in CS[1:]:
        base_pu = np.array([np.mean(x[f'c{CS[0]}_pu']) for x in rows if x[f'c{CS[0]}_pu'] and x[f'c{c}_pu']])
        cur_pu = np.array([np.mean(x[f'c{c}_pu']) for x in rows if x[f'c{CS[0]}_pu'] and x[f'c{c}_pu']])
        dd = cur_pu - base_pu
        w = wilcoxon(dd)
        print(f'c={c} vs c=0 (预测类未观察): Δ={dd.mean():+.4f} 正港{int((dd>0).sum())}/{len(dd)} p={w.pvalue:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
