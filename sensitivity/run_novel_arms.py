"""四候选新颖改造臂（NOVEL_ARMS_PREREG_v1）。用法：python run_novel_arms.py --mode m1_interval|m2_conflict|m3_gated|m4_harm [--smoke]
单点改造 + 全链；用 run_adaptive 的参数化副本（tau 向量/伪计数向量/分段供给/伤害预算）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from bdh_pixel_loss_controls import conditional_likelihood as lik
from bdh_pixel_loss_controls.product_correlated_source import fixed_count
from bdh_pixel_loss_controls.few_label_transfer import utility_mass
from bdh_pixel_loss_controls.utility_transport import divide
from run_adaptive import (CACHE, project_class_evidence_rt, natural_policy, macro_ba,
                          src_weights, SEEDS, ORDINALS, DRAWS, K, TAU0,
                          shared_cov_predict_tauvec, class_mass_pseudovec)

OUT = Path(r'E:/Hermes/paper_tracking/sensitivity/novel_arms')
OUT.mkdir(parents=True, exist_ok=True)
EPSILON = 1e-12


def interval_tau(source_pos, ports_all, y_ids, annotation_means, z_ann):
    """m1_interval：港间逐维 [min,med,max]；τ_c = τ(1+β·d/w)。"""
    tau = np.full(K, TAU0)
    for c in range(K):
        mask = y_ids[source_pos] == c
        if mask.sum() < 2:
            continue
        pm = []
        for p in np.unique(ports_all[source_pos[mask]]):
            pm.append(source_pos[mask & (ports_all[source_pos] == p)].mean(0))
        pm = np.vstack(pm)
        lo, hi = pm.min(0), pm.max(0)
        width = np.maximum(hi - lo, 1e-6)
        d = np.where(annotation_means[c] < lo, lo - annotation_means[c],
                     np.where(annotation_means[c] > hi, annotation_means[c] - hi, 0.0))
        ratio = (d / width).mean()
        tau[c] = float(np.clip(TAU0 * (1 + 0.5 * ratio), TAU0, 3 * TAU0))
    return tau


def conflict_tau(source_model, annotation_means):
    """m2_conflict：Mahalanobis(标注均值, 源均值; Σ) 调 τ。"""
    tau = np.full(K, TAU0)
    for c in range(K):
        delta = annotation_means[c] - source_model['means'][c]
        d = float(np.sqrt(delta @ source_model['precision'] @ delta))
        tau[c] = float(TAU0 * (1 + min(d, 4.0)))
    return tau


def run(mode, smoke=False):
    out = {'ports': {}}
    for port in sorted(p.name[:-4] for p in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{port}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        base_logits = d['base_logits']; expert_logits = d['expert_logits']
        feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
        qpos = d['qpos']; spos = d['spos']; ranking = d['ranking']
        base = base_logits.argmax(1); expert = expert_logits.argmax(1)
        query_n = len(qpos)
        source_n = min(query_n, len(spos))
        records = []
        for ordinal in ORDINALS:
            rng_s = np.random.default_rng(1000 + ordinal)
            strat = {}
            for c in np.unique(y_ids[spos]):
                pos = spos[y_ids[spos] == c]
                strat[c] = rng_s.choice(pos, max(1, int(round(source_n * len(pos) / len(spos)))), replace=False)
            source_pos = np.concatenate(list(strat.values()))
            source_model = lik.fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
            cond = np.bincount(K * K * y_ids[source_pos] + K * base[source_pos] + expert[source_pos],
                               weights=src_weights(y_ids[source_pos], ports_all[source_pos]),
                               minlength=K ** 3).reshape(K, K, K)
            cond = cond / cond.sum(axis=(1, 2))[:, None, None]
            for seed in SEEDS:
                for draw in range(DRAWS):
                    rng = np.random.default_rng(seed * 1000 + ordinal * 100 + draw)
                    ann = rng.choice(qpos, 64, replace=False)
                    ay = y_ids[ann]
                    observed = np.bincount(ay, minlength=K) > 0
                    priority = rng.permutation(query_n)
                    b, e = base[qpos], expert[qpos]
                    s_flat = divide(utility_mass(cond, observed, True),
                                    np.bincount(K * b + e, minlength=K * K).reshape(K, K) / len(b))[b, e]
                    z_ann = (feats[ann] - source_model['center']) / source_model['scale']
                    ann_means = np.zeros((K, 14))
                    for c in np.flatnonzero(np.bincount(ay, minlength=K)):
                        ann_means[c] = z_ann[ay == c].mean(0)
                    tau_vec = np.full(K, TAU0)
                    if mode == 'm1_interval':
                        tau_vec = interval_tau(source_pos, ports_all, y_ids, ann_means, z_ann)
                    elif mode == 'm2_conflict':
                        tau_vec = conflict_tau(source_model, ann_means)
                    pseudo_vec = np.full(K, 0.5)
                    p1 = shared_cov_predict_tauvec(source_model, feats[ann], ay, feats[qpos], tau_vec)
                    f1 = class_mass_pseudovec(p1['loglikelihood'], p1['counts'], pseudo_vec)
                    ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
                    if mode == 'm3_gated':
                        high = ratio2.max(1) > np.median(ratio2.max(1))
                        pseudo_g = p1['counts'] + 0.5
                        prior_g = np.zeros(K)
                        for c in range(K):
                            prior_g[c] = pseudo_g[c] + f1['posterior'][high, c].sum()
                        prior_g = prior_g / prior_g.sum()
                        ratio2 = project_class_evidence_rt(p1['loglikelihood'], prior_g)
                    s2 = lik.robust_scores(ratio2, b, e, observed)
                    arms = {'flat': s_flat, 'class_projection': s2}
                    natural = {name: natural_policy(b, e, s, ranking[qpos], priority, .05)
                               for name, s in arms.items()}
                    if mode == 'm4_harm':
                        harm = np.maximum(0.0, -s2)
                        H = 0.1 * np.maximum(0.0, s2).sum()
                        candidates = np.flatnonzero((b != e) & (s2 > EPSILON))
                        order = candidates[np.lexsort((priority[candidates], -ranking[qpos][candidates],
                                                      -s2[candidates]))]
                        cap = int(np.clip(round(0.05 * query_n), 1, len(order)))  # 等预算：与冻结同 5%
                        cum = np.cumsum(harm[order])
                        k = int((cum <= H).sum())
                        if k == 0:
                            k = int(np.clip(round(0.05 * query_n), 1, len(order)))
                        mask = np.zeros(query_n, dtype=bool)
                        mask[order[:min(k, cap)]] = True
                        natural['class_projection']['selected'] = mask
                    kk = min(int(natural[name]['selected'].sum()) for name in arms)
                    matched = {name: fixed_count(b, e, arms[name], ranking[qpos], priority, kk)
                               for name in arms}
                    y_q = y_ids[qpos]
                    for name, s in (('flat', s_flat), ('class_projection', s2)):
                        for mode_, mask in (('natural', natural[name]['selected']), ('matched', matched[name])):
                            pred = np.where(mask, expert[qpos], base[qpos])
                            records.append(dict(arm=name, mode=mode_, ba=macro_ba(y_q, pred)))
        rec = pd.DataFrame(records)
        per = rec.groupby(['arm', 'mode']).ba.agg('mean').reset_index()
        bf = per[(per['arm'] == 'flat') & (per['mode'] == 'matched')]['ba'].iloc[0]
        v = per[(per['arm'] == 'class_projection') & (per['mode'] == 'matched')]['ba'].iloc[0]
        nat = float(per[(per['arm'] == 'class_projection') & (per['mode'] == 'natural')]['ba'].iloc[0]
                    - per[(per['arm'] == 'flat') & (per['mode'] == 'natural')]['ba'].iloc[0])
        out['ports'][port] = {'n': int(d['n']), 'cp_flat': v - bf, 'natural': nat}
        print(f'{mode} {port}: {100*(v-bf):+.3f}pp', flush=True)
        if smoke:
            break
    plist = [p for p in out['ports'] if 'cp_flat' in out['ports'][p]]
    cp = np.array([out['ports'][p]['cp_flat'] for p in plist])
    nat = np.array([out['ports'][p]['natural'] for p in plist])
    res = {'mode': mode, 'n_ports': len(plist), 'cp_flat_mean': float(cp.mean()),
           'cp_flat_positive': int((cp > 0).sum()), 'cp_flat_worst': float(cp.min()),
           'natural_mean': float(nat.mean())}
    (OUT / f'summary_{mode}.json').write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
    (OUT / f'result_{mode}.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    run(a.mode, smoke=a.smoke)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
