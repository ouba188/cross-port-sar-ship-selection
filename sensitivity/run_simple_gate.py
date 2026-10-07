"""验证：简单门控（专家预测类已观察 + 似然更高）vs 完整 s2。
独立 Student-t 适配（shared_cov_predict），同一模型/标注/eval/seed。
简单门控：switch = D & observed[eq] & (ll[eq] > ll[bq])，否则 base。
用法：python run_simple_gate.py [--smoke]
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


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


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
    return robust_scores(ratio2, bq, eq, observed), p1['loglikelihood'], observed


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
        row = {'port': p, 'sg': [], 's2': []}
        for seed in SEEDS:
            ann = np.random.default_rng(seed).choice(qpos, ANNOT, replace=False)
            ay = y_ids[ann]
            in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
            ev = ~in_ann
            s2, ll, observed = full_s2(feats, y_ids, ports_all, qpos, spos, base, expert, ann)
            # 简单门控：专家预测类已观察 且 似然更高 → 专家
            switch = D & observed[eq] & (ll[np.arange(query_n), eq] > ll[np.arange(query_n), bq])
            g_sg = np.where(switch, eq, bq)
            row['sg'].append(macro_ba(yq[ev], g_sg[ev]))
            # 完整 s2
            g_s2 = np.where(D, np.where(s2 > 0, eq, bq), bq)
            row['s2'].append(macro_ba(yq[ev], g_s2[ev]))
        rows.append(row)
        print(f'{p}: 简单门控={np.mean(row["sg"]):.4f} s2={np.mean(row["s2"]):.4f} (Δ={np.mean(row["s2"])-np.mean(row["sg"]):+.4f})', flush=True)
        if smoke:
            break
    json.dump(rows, open('simple_gate.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    from scipy.stats import wilcoxon
    sg = np.array([np.mean(x['sg']) for x in rows])
    s2 = np.array([np.mean(x['s2']) for x in rows])
    dd = s2 - sg
    w = wilcoxon(dd)
    print(f'\n简单门控 BA={sg.mean():.6f}')
    print(f'完整 s2  BA={s2.mean():.6f}')
    print(f's2 − 简单门控 = {dd.mean():+.6f} ({(dd.mean()*100):+.4f}pp) 正港{int((dd>0).sum())}/{len(dd)} p={w.pvalue:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
