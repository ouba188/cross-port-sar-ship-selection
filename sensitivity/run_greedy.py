"""robust_scores 贪心替换（GREEDY_KSCALE_PREREG_v1）：全链复现验证。
与冻结链唯一差异 = robust_scores → greedy_min_mean_superset。
"""
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

OUT = Path(r'E:/Hermes/paper_tracking/sensitivity/novel_arms/greedy')
OUT.mkdir(parents=True, exist_ok=True)
EPSILON = 1e-12


def greedy_min_mean_superset(ratio, base, expert, observed):
    """最小均值超集精确贪心：S⊇observed，(Σ_{c∈S} a_c)/|S|，a_c=r·1[c=expert]-h·1[c=base]。"""
    ratio, base, expert, observed = map(np.asarray, (ratio, base, expert, observed))
    n, K = ratio.shape
    r = ratio[np.arange(n), expert]
    h = ratio[np.arange(n), base]
    O = np.flatnonzero(observed)
    m0 = len(O)
    A0 = np.zeros(n)
    for c in O:
        A0 += np.where(expert == c, r, 0.0) - np.where(base == c, h, 0.0)
    out = np.empty(n)
    comp_idx = np.array([c for c in range(K) if c not in O])
    for i in range(n):
        A = A0[i]; m = m0
        vals = sorted((r[i] if c == expert[i] else 0.0) - (h[i] if c == base[i] else 0.0)
                      for c in comp_idx)
        mu = A / m
        for x in vals:
            if x < mu:
                A += x; m += 1; mu = A / m
            else:
                break
        out[i] = mu
    return out


def run(smoke=False):
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
                    s2 = greedy_min_mean_superset(ratio2, b, e, observed)   # 唯一差异
                    arms = {'flat': s_flat, 'class_projection': s2}
                    natural = {name: natural_policy(b, e, s, ranking[qpos], priority, .05)
                               for name, s in arms.items()}
                    kk = min(int(natural[name]['selected'].sum()) for name in arms)
                    matched = {name: fixed_count(b, e, arms[name], ranking[qpos], priority, kk)
                               for name in arms}
                    y_q = y_ids[qpos]
                    for name, s in arms.items():
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
        print(f'greedy {port}: {100*(v-bf):+.3f}pp', flush=True)
        if smoke:
            break
    plist = [p for p in out['ports'] if 'cp_flat' in out['ports'][p]]
    cp = np.array([out['ports'][p]['cp_flat'] for p in plist])
    nat = np.array([out['ports'][p]['natural'] for p in plist])
    res = {'mode': 'greedy', 'n_ports': len(plist), 'cp_flat_mean': float(cp.mean()),
           'cp_flat_positive': int((cp > 0).sum()), 'cp_flat_worst': float(cp.min()),
           'natural_mean': float(nat.mean())}
    (OUT / 'summary.json').write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
