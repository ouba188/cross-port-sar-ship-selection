"""改法 1：目标侧温度校准置信度 + s2 组合（可靠性信号）。
s2=类证据键；conf=专家置信度（温度 T 在 64 标签上拟合，修正"值不迁移"）。
ensemble = zscore(s2) + zscore(conf)，双向符号门控。对比 s2 alone / s2+raw / s2+cal。
用法：python run_calibrated_conf.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from bdh_pixel_loss_controls.conditional_likelihood import fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import project_class_evidence_rt, K

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
ANNOT = 64
SEEDS = [0, 1, 2, 3, 4]


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def wacc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def fit_temperature(logits_ann, y_ann):
    """温度缩放：T* 最小化 64 标签上的 NLL。"""
    def nll(T):
        T = max(T, 1e-2)
        p = softmax(logits_ann / T, axis=1)
        return -float(np.log(np.clip(p[np.arange(len(y_ann)), y_ann], 1e-9, 1)).mean())
    res = minimize_scalar(nll, bounds=(0.05, 10.0), method='bounded')
    return float(res.x)


def s2_key(feats, y_ids, ports_all, qpos, spos, base, expert, ann):
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
        expert_logits = d['expert_logits']
        yq = y_ids[qpos]; bq = base[qpos]; eq = expert[qpos]
        query_n = len(qpos)
        row = {'port': p, 's2_ba': [], 's2raw_ba': [], 's2cal_ba': [], 's2_acc': [], 's2cal_acc': []}
        for seed in SEEDS:
            ann = np.random.default_rng(seed).choice(qpos, ANNOT, replace=False)
            ay = y_ids[ann]
            in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
            ev = ~in_ann
            s2 = s2_key(feats, y_ids, ports_all, qpos, spos, base, expert, ann)
            T = fit_temperature(expert_logits[ann], ay)
            conf_raw = softmax(expert_logits[qpos], axis=1).max(1)
            conf_cal = softmax(expert_logits[qpos] / T, axis=1).max(1)
            D = (bq != eq)
            def zs(x):
                return (x - x.mean()) / (x.std() + 1e-9)
            g_s2 = np.where(D, np.where(s2 > 0, eq, bq), bq)
            g_raw = np.where(D, np.where(zs(s2) + zs(conf_raw) > 0, eq, bq), bq)
            g_cal = np.where(D, np.where(zs(s2) + zs(conf_cal) > 0, eq, bq), bq)
            row['s2_ba'].append(macro_ba(yq[ev], g_s2[ev]))
            row['s2raw_ba'].append(macro_ba(yq[ev], g_raw[ev]))
            row['s2cal_ba'].append(macro_ba(yq[ev], g_cal[ev]))
            row['s2_acc'].append(wacc(yq[ev], g_s2[ev]))
            row['s2cal_acc'].append(wacc(yq[ev], g_cal[ev]))
        rows.append(row)
        print(f'{p}: s2={np.mean(row["s2_ba"]):.4f} s2+raw={np.mean(row["s2raw_ba"]):.4f} '
              f's2+cal={np.mean(row["s2cal_ba"]):.4f}', flush=True)
        if smoke:
            break

    json.dump(rows, open('calibrated_conf.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for k in ['s2_ba', 's2raw_ba', 's2cal_ba']:
        v = np.array([np.mean(x[k]) for x in rows])
        print(f'{k}: BA={v.mean():.4f}')
    s2 = np.array([np.mean(x['s2_ba']) for x in rows]); cal = np.array([np.mean(x['s2cal_ba']) for x in rows])
    from scipy.stats import wilcoxon
    d = cal - s2
    w = wilcoxon(d)
    print(f's2+cal vs s2: Δ={d.mean():+.4f} 正港{int((d>0).sum())}/{len(d)} p={w.pvalue:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
