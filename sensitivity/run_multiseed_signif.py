"""多 seed 稳定性 + paired 显著性（s2 双向 vs fusion）。统一口径 query\ann。
用法：python run_multiseed_signif.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits
from scipy.stats import wilcoxon

from bdh_pixel_loss_controls.conditional_likelihood import fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import project_class_evidence_rt, K

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
SEEDS = [0, 1, 2, 3, 4]
ANNOT = 64


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def wacc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def s2_bidir(feats, y_ids, ports_all, qpos, spos, base, expert, ann):
    """给定 64 标签 ann，算 s2 键 + 双向选择预测（分歧样本 s2>0→expert）。"""
    query_n = len(qpos); source_n = min(query_n, len(spos))
    rng_s = np.random.default_rng(1000)
    strat = {c: rng_s.choice(spos[y_ids[spos] == c],
                             max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False)
             for c in np.unique(y_ids[spos])}
    source_pos = np.concatenate(list(strat.values()))
    ay = y_ids[ann]
    bq = base[qpos]; eq = expert[qpos]
    observed = np.bincount(ay, minlength=K) > 0
    sm = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
    p1 = shared_cov_predict(sm, feats[ann], ay, feats[qpos])
    f1 = class_mass(p1['loglikelihood'], p1['counts'])
    ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
    s2 = robust_scores(ratio2, bq, eq, observed)
    D = (bq != eq)
    return np.where(D, np.where(s2 > 0, eq, bq), bq)


def run(smoke=False):
    rows = []  # 每行: {port, s2_ba_seeds:[], s2_acc_seeds:[], fusion_ba_seeds:[], fusion_acc_seeds:[]}
    for p in sorted(x.name[:-4] for x in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
        qpos = d['qpos']; spos = d['spos']
        base = d['base_logits'].argmax(1); expert = d['expert_logits'].argmax(1)
        yq = y_ids[qpos]; eq = expert[qpos]
        query_n = len(qpos)
        row = {'port': p, 's2_ba': [], 's2_acc': [], 'fu_ba': [], 'fu_acc': []}
        for seed in SEEDS:
            ann = np.random.default_rng(seed).choice(qpos, ANNOT, replace=False)
            in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
            ev = ~in_ann
            bidir = s2_bidir(feats, y_ids, ports_all, qpos, spos, base, expert, ann)
            row['s2_ba'].append(macro_ba(yq[ev], bidir[ev]))
            row['s2_acc'].append(wacc(yq[ev], bidir[ev]))
            row['fu_ba'].append(macro_ba(yq[ev], eq[ev]))
            row['fu_acc'].append(wacc(yq[ev], eq[ev]))
        rows.append(row)
        print(f'{p}: s2_BA={np.mean(row["s2_ba"]):.4f}±{np.std(row["s2_ba"]):.4f}  '
              f'fu_BA={np.mean(row["fu_ba"]):.4f}  Δ={np.mean(row["s2_ba"])-np.mean(row["fu_ba"]):+.4f}', flush=True)
        if smoke:
            break

    json.dump(rows, open('multiseed_signif.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    s2_ba = np.array([np.mean(r['s2_ba']) for r in rows])
    fu_ba = np.array([np.mean(r['fu_ba']) for r in rows])
    s2_acc = np.array([np.mean(r['s2_acc']) for r in rows])
    fu_acc = np.array([np.mean(r['fu_acc']) for r in rows])
    print(f'\ns2  BA={s2_ba.mean():.4f} (跨seed std 均值 {np.mean([np.std(r["s2_ba"]) for r in rows]):.4f})  '
          f'ACC={s2_acc.mean():.4f}')
    print(f'fu  BA={fu_ba.mean():.4f}  ACC={fu_acc.mean():.4f}')
    d = s2_ba - fu_ba
    w = wilcoxon(d)
    print(f'paired 差异 BA: mean={d.mean():+.4f}, 正港 {int((d > 0).sum())}/23, '
          f'Wilcoxon p={w.pvalue:.4f}')
    # bootstrap 95% CI of mean difference
    rng = np.random.default_rng(42)
    boots = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(5000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f'bootstrap 95% CI of mean ΔBA: [{lo:+.4f}, {hi:+.4f}]')
    print('判据: ', 'PASS' if (w.pvalue < 0.05 and lo > 0) else 'FAIL')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
