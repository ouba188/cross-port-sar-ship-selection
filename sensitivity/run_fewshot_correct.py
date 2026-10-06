"""路由 + 少样本修正 M5（FEWSHOT_CORRECT_PREREG_v1）。5 臂同协议。
用法：python run_fewshot_correct.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.linear_model import Ridge

from bdh_pixel_loss_controls.conditional_likelihood import fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import project_class_evidence_rt, K, natural_policy

OUT = Path(r'E:/Hermes/paper_tracking/sensitivity')
CACHE = OUT / 'pred_cache'
CACHE2 = OUT / 'pred_cache2'
ANNOT = 64
BUDGET = 0.05


def macro_ba(y, pred):
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def wacc(y, pred):
    return float((np.asarray(y) == np.asarray(pred)).mean())


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def run(smoke=False):
    rows = []
    for p in sorted(x.name[:-4] for x in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        d2 = np.load(CACHE2 / f'{p}.npz', allow_pickle=False)
        Xs, Z = d2['Xs'], d2['Z']
        feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
        qpos = d['qpos']; spos = d['spos']
        base = d['base_logits'].argmax(1); expert = d['expert_logits'].argmax(1)
        expert_logits = d['expert_logits']
        query_n = len(qpos); source_n = min(query_n, len(spos))
        rng_s = np.random.default_rng(1000)
        strat = {c: rng_s.choice(spos[y_ids[spos] == c],
                                 max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False)
                 for c in np.unique(y_ids[spos])}
        source_pos = np.concatenate(list(strat.values()))
        yq = y_ids[qpos]; bq = base[qpos]; eq = expert[qpos]
        X = np.hstack([Xs, Z])

        # 64 标签 + fewshot 分类器（[VV|VH]）
        ann = np.random.default_rng(0).choice(qpos, ANNOT, replace=False)
        ay = y_ids[ann]
        _, inv, cnt = np.unique(ay, return_inverse=True, return_counts=True)
        sw = 1.0 / (len(cnt) * cnt[inv])
        m_fs = Ridge(alpha=1.0).fit(X[ann], np.eye(K)[ay], sample_weight=sw)
        fs_logits = m_fs.predict(X[qpos])
        fs = fs_logits.argmax(1)

        # 路由键 s2（冻结 M4）
        source_model = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
        observed = np.bincount(ay, minlength=K) > 0
        p1 = shared_cov_predict(source_model, feats[ann], ay, feats[qpos])
        f1 = class_mass(p1['loglikelihood'], p1['counts'])
        ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
        s2 = robust_scores(ratio2, bq, eq, observed)
        priority = np.random.default_rng(2).permutation(query_n)
        nat = natural_policy(bq, eq, s2, d['ranking'][qpos], priority, BUDGET)
        routed = nat['selected']

        # 臂
        oth = np.where(routed, eq, bq)                       # A
        m5_fs = np.where(routed, fs, bq)                     # B
        pe = softmax(expert_logits[qpos]); pf = softmax(fs_logits)
        blend = (pe + pf).argmax(1)                          # C
        m5_blend = np.where(routed, blend, bq)

        in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
        ev = ~in_ann
        arms = {'OTH': oth, 'M5-fewshot': m5_fs, 'M5-blend': m5_blend,
                'fewshot-full': fs, 'fusion': eq}
        row = {'port': p, 'n': int(d['n'])}
        for a, pred in arms.items():
            row[f'ba_{a}'] = macro_ba(yq[ev], pred[ev])
            row[f'acc_{a}'] = wacc(yq[ev], pred[ev])
        rows.append(row)
        print(f'{p}: ' + ' '.join(f'{a}={row[f"ba_{a}"]:.3f}' for a in arms), flush=True)
        if smoke:
            break

    (OUT / 'fewshot_correct.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    arms = ['OTH', 'M5-fewshot', 'M5-blend', 'fewshot-full', 'fusion']
    for a in arms:
        ba = np.array([r[f'ba_{a}'] for r in rows]); acc = np.array([r[f'acc_{a}'] for r in rows])
        print(f'{a:12s} BA={ba.mean():.4f} (pos {(ba > 0).sum()}/{len(ba)})  ACC={acc.mean():.4f}')
    print('判据: M5 BA > OTH BA?',
          float(np.mean([r['ba_M5-fewshot'] for r in rows])) > float(np.mean([r['ba_OTH'] for r in rows])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
