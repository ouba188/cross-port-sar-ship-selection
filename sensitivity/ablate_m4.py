"""M4 消融臂：class_projection 证据下 排序选 vs 同规模随机选（链内口径）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from bdh_pixel_loss_controls import conditional_likelihood as lik
from bdh_pixel_loss_controls import shared_covariance_likelihood as shared_cov
from bdh_pixel_loss_controls.product_correlated_source import fixed_count
from bdh_pixel_loss_controls.few_label_transfer import utility_mass
from bdh_pixel_loss_controls.utility_transport import divide
from run_adaptive import (CACHE, project_class_evidence_rt, natural_policy, macro_ba,
                          src_weights, SEEDS, ORDINALS, DRAWS, K)

OUT = Path(r'E:/Hermes/paper_tracking/sensitivity')
RES = OUT / 'm4_ablation.json'


def run():
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
                    p1 = shared_cov.predict(source_model, feats[ann], ay, feats[qpos])
                    f1 = lik.class_mass(p1['loglikelihood'], p1['counts'])
                    ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
                    s2 = lik.robust_scores(ratio2, b, e, observed)
                    natural = natural_policy(b, e, s2, ranking[qpos], priority, .05)
                    kk = int(natural['selected'].sum())
                    ranked = fixed_count(b, e, s2, ranking[qpos], priority, kk)
                    eligible = np.flatnonzero(b != e)
                    random = np.zeros(len(b), dtype=bool)
                    random[rng.choice(eligible, min(kk, len(eligible)), replace=False)] = True
                    y_q = y_ids[qpos]
                    for name, mask in (('ranked', ranked), ('random', random), ('natural', natural['selected'])):
                        pred = np.where(mask, expert[qpos], base[qpos])
                        records.append(dict(arm=name, ba=macro_ba(y_q, pred)))
        rec = pd.DataFrame(records)
        per = rec.groupby('arm').ba.mean()
        ranked_ba = per['ranked']; random_ba = per['random']
        out['ports'][port] = {'n': int(d['n']), 'ranked_ba': ranked_ba, 'random_ba': random_ba,
                              'rank_minus_random': ranked_ba - random_ba}
        print(f'{port}: ranked {ranked_ba:.4f} random {random_ba:.4f} Δ {100*(ranked_ba-random_ba):+.3f}pp', flush=True)
    deltas = np.array([v['rank_minus_random'] for v in out['ports'].values()])
    res = {'n_ports': len(deltas), 'rank_minus_random_mean': float(deltas.mean()),
           'positive_ports': int((deltas > 0).sum()), 'worst': float(deltas.min())}
    RES.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
