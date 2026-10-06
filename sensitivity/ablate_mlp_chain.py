"""预测器替换对照臂 v2：MLP 当 base/expert 进 M1→M4 链（预注册：只报告，不进主线）。
base=MLP(Xs)、expert=MLP([Xs|Z])；plain-CE、宽 256×2、类权重平衡、CPU。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from threadpoolctl import threadpool_limits

from bdh_pixel_loss_controls import conditional_likelihood as lik
from bdh_pixel_loss_controls import shared_covariance_likelihood as shared_cov
from bdh_pixel_loss_controls.product_correlated_source import fixed_count
from bdh_pixel_loss_controls.few_label_transfer import utility_mass
from bdh_pixel_loss_controls.utility_transport import divide
from run_adaptive import (CACHE, project_class_evidence_rt, natural_policy, macro_ba,
                          src_weights, SEEDS, ORDINALS, DRAWS, K)

CACHE2 = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache2')
OUT = Path(r'E:/Hermes/paper_tracking/sensitivity')
RES = OUT / 'mlp_chain.json'


def mlp_logits(Xtr, ytr, Xall):
    d = Xtr.shape[1]
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Xall_t = torch.tensor(Xall, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    counts = np.bincount(ytr, minlength=K).astype(np.float64)
    w = (counts.sum() / (K * counts)).astype(np.float32)
    model = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                          nn.Linear(256, K))
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w))
    for _ in range(30):
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), 128):
            idx = perm[i:i + 128]
            out = model(Xtr_t[idx])
            loss = crit(out, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model(Xall_t).numpy()


def run():
    out = {'ports': {}}
    for port in sorted(p.name[:-4] for p in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{port}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        d2 = np.load(CACHE2 / f'{port}.npz')
        Xs, Z = d2['Xs'], d2['Z']
        feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
        qpos = d['qpos']; spos = d['spos']; ranking = d['ranking']
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
            lb = mlp_logits(Xs[spos], y_ids[spos], Xs)
            le = mlp_logits(np.c_[Xs[spos], Z[spos]], y_ids[spos], np.c_[Xs, Z])
            gb = lb.argmax(1); ge = le.argmax(1)
            cond = np.bincount(K * K * y_ids[source_pos] + K * gb[source_pos] + ge[source_pos],
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
                    b, e = gb[qpos], ge[qpos]
                    s_flat = divide(utility_mass(cond, observed, True),
                                    np.bincount(K * b + e, minlength=K * K).reshape(K, K) / len(b))[b, e]
                    p1 = shared_cov.predict(source_model, feats[ann], ay, feats[qpos])
                    f1 = lik.class_mass(p1['loglikelihood'], p1['counts'])
                    ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
                    s2 = lik.robust_scores(ratio2, b, e, observed)
                    natural = natural_policy(b, e, s2, ranking[qpos], priority, .05)
                    kk = int(natural['selected'].sum())
                    matched = fixed_count(b, e, s2, ranking[qpos], priority, kk)
                    y_q = y_ids[qpos]
                    for name, mask in (('flat_utility', np.zeros(len(b), dtype=bool)),
                                       ('class_projection', matched)):
                        pred = np.where(mask, e, b)
                        records.append(dict(arm=name, ba=macro_ba(y_q, pred)))
        rec = pd.DataFrame(records)
        per = rec.groupby('arm').ba.mean()
        out['ports'][port] = {'n': int(d['n']), 'mlp_base_ba': per['flat_utility'],
                              'mlp_cp_ba': per['class_projection'],
                              'cp_minus_base': per['class_projection'] - per['flat_utility']}
        print(f'{port}: base {per["flat_utility"]:.4f} cp {per["class_projection"]:.4f} '
              f'Δ {100*(per["class_projection"]-per["flat_utility"]):+.3f}pp', flush=True)
    deltas = np.array([v['cp_minus_base'] for v in out['ports'].values()])
    base_bas = np.array([v['mlp_base_ba'] for v in out['ports'].values()])
    res = {'n_ports': len(deltas), 'mlp_base_ba_mean': float(base_bas.mean()),
           'cp_minus_base_mean': float(deltas.mean()), 'positive_ports': int((deltas > 0).sum()),
           'worst': float(deltas.min())}
    RES.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
