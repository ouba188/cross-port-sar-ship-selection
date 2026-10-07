"""主线链参数敏感性（SENSITIVITY_PREREG_v1）。预测器/M1 冻结缓存复用，只重算 M2/M3/M4。
用法：python sweep_sensitivity.py --param tau --value 4 [--smoke]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from bdh_pixel_loss_controls import conditional_likelihood as lik
from bdh_pixel_loss_controls import shared_covariance_likelihood as shared_cov
from bdh_pixel_loss_controls import rank_profile_transport
from bdh_pixel_loss_controls.product_correlated_source import fixed_count
from bdh_capacity_evidence import retrieve
from bdh_pixel_loss_controls.few_label_transfer import utility_mass
from bdh_pixel_loss_controls.utility_transport import divide
from report_bdh_pixel_handoff import budget_policy

WS = Path(r'E:/临时会话/visual_reliable_baseline')
ROOT = WS / 'artifacts/eight_class_adaptive_20260916'
OUT = Path(r'E:/Hermes/paper_tracking/sensitivity')
CACHE = OUT / 'pred_cache'
CACHE.mkdir(parents=True, exist_ok=True)
SEEDS = (20261005, 20261006)
ORDINALS = (0, 1)
DRAWS = 20
K = 8
EPSILON = 1e-12


def macro_ba(y, pred):
    present = np.unique(y)
    return float(np.mean([(pred[y == c] == c).mean() for c in present]))


def natural_policy(base, expert, utility, ranking, priority, frac):
    """rank_profile_transport.policy 的参数化同构副本（唯一改动：.05 -> frac + 白名单移除）。
    与 budget_policy 在 0.05/0.10/0.20 逐位一致（0.05 已复现 +1.371pp 验证）。"""
    base, expert, utility, ranking, priority = map(np.asarray, (base, expert, utility, ranking, priority))
    eligible = (base != expert) & (utility > EPSILON)
    cap = min(len(base), max(1, round(frac * len(base))))
    adm_order = np.lexsort((priority, -utility))
    admitted = adm_order[eligible[adm_order]][:cap]
    candidates = np.flatnonzero((base != expert) & (utility > EPSILON))
    order = candidates[np.lexsort((priority[candidates], -ranking[candidates], -utility[candidates]))]
    limit = len(admitted)
    mask = np.zeros(len(base), dtype=bool)
    mask[order[:limit]] = True
    return dict(selected=mask, cap=cap)


def project_class_evidence_rt(loglikelihood, prior):
    cost = -np.asarray(loglikelihood, dtype=np.float64)
    try:
        retrieval = retrieve(cost, prior, 'capacity')
    except ValueError:
        import ot
        shifted = cost - cost.min(1, keepdims=True)
        coupling = ot.sinkhorn(np.full(len(cost), 1 / len(cost)), prior, shifted,
                               reg=1., numItermax=50000, stopThr=1e-11)
        weights = len(cost) * coupling
        if float(np.abs(weights.mean(0) - prior).max()) > 1e-8:
            raise RuntimeError('solver failed')
        retrieval = dict(weights=weights)
    return retrieval['weights'] / prior[None]


def build_cache():
    """首跑：从冻结协议重建每折预测器与 M1 输入并缓存（与 internal_mainline 同构）。"""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeClassifier
    rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8-sig')))
    R = np.load(ROOT / 'pooled_features.npy', mmap_mode='r').astype(np.float32)
    S = np.load(WS / 'b2_ssl_feats_38091.npz')['F'].astype(np.float32)
    ports_all = np.array([r['port'] for r in rows])
    y_ids = np.array([int(r['class_id']) for r in rows])
    mmsi = np.array([r['mmsi'] or '' for r in rows])
    product = np.array([r.get('product') or r.get('product_id') or '' for r in rows])
    for port in sorted(set(ports_all.tolist())):
        cache = CACHE / f'{port}.npz'
        if cache.exists():
            continue
        te_mask = ports_all == port
        if int(te_mask.sum()) <= 64:
            np.savez_compressed(cache, skipped=True, n=int(te_mask.sum()))
            continue
        tr_mask = ~te_mask
        q_mmsi = set(mmsi[te_mask]) - {''}
        q_prod = set(product[te_mask]) - {''}
        tr_mask = tr_mask & ~np.isin(mmsi, list(q_mmsi)) & ~np.isin(product, list(q_prod))
        pca = PCA(n_components=128, random_state=0).fit(R[tr_mask])
        X = pca.transform(R)
        mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0) + 1e-6
        Xs = (X - mu) / sd
        mb = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Xs[tr_mask], y_ids[tr_mask])
        base_logits = mb.decision_function(Xs)
        pca2 = PCA(n_components=64, random_state=0).fit(S[tr_mask])
        Z = pca2.transform(S)
        Mtr = np.c_[Xs[tr_mask], Z[tr_mask]]
        Mall = np.c_[Xs, Z]
        mu2, sd2 = Mtr.mean(0), Mtr.std(0) + 1e-6
        me = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Mtr - mu2) / sd2, y_ids[tr_mask])
        expert_logits = me.decision_function((Mall - mu2) / sd2)
        base = base_logits.argmax(1); expert = expert_logits.argmax(1)
        pb = np.exp(base_logits - base_logits.max(1, keepdims=True)); pb /= pb.sum(1, keepdims=True)
        pe = np.exp(expert_logits - expert_logits.max(1, keepdims=True)); pe /= pe.sum(1, keepdims=True)
        feats = lik.decision_features(base_logits, expert_logits)
        qpos = np.flatnonzero(te_mask)
        spos = np.flatnonzero(tr_mask)
        np.savez_compressed(cache, n=int(te_mask.sum()), base_logits=base_logits.astype(np.float32),
                            expert_logits=expert_logits.astype(np.float32), feats=feats,
                            y_ids=y_ids, ports_all=ports_all, qpos=qpos, spos=spos,
                            ranking=(pe.max(1) - pb.max(1)))
        print('cache', port, flush=True)
    print('cache done', flush=True)


def run_config(param, value, smoke=False):
    frac = 0.05
    tau, pseudo, em, n_annot = 8., 0.5, 128, 64
    if param == 'tau':
        tau = float(value)
    elif param == 'pseudo':
        pseudo = float(value)
    elif param == 'em':
        em = int(value)
    elif param == 'annot':
        n_annot = int(value)
    elif param == 'budget':
        frac = float(value)
    lik.TAU = tau
    lik.PSEUDOCOUNT = pseudo
    lik.EM_STEPS = em
    shared_cov.TAU = tau
    ports_list = sorted(p.name[:-4] for p in CACHE.glob('*.npz'))
    out = {'ports': {}}
    for port in ports_list:
        d = np.load(CACHE / f'{port}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            out['ports'][port] = {'n': int(d['n']), 'skipped_small': True}
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
                    if query_n <= n_annot:
                        continue
                    ann = rng.choice(qpos, n_annot, replace=False)
                    ay = y_ids[ann]
                    observed = np.bincount(ay, minlength=K) > 0
                    priority = rng.permutation(query_n)
                    b, e = base[qpos], expert[qpos]
                    s_flat = divide(utility_mass(cond, observed, True),
                                    np.bincount(K * b + e, minlength=K * K).reshape(K, K) / len(b))[b, e]
                    p0 = lik.predictive_likelihood(source_model, feats[ann], ay, feats[qpos], False)
                    f0 = lik.class_mass(p0['loglikelihood'], p0['counts'])
                    s0 = lik.robust_scores(f0['ratio'], b, e, observed)
                    p1 = shared_cov.predict(source_model, feats[ann], ay, feats[qpos])
                    f1 = lik.class_mass(p1['loglikelihood'], p1['counts'])
                    s1 = lik.robust_scores(f1['ratio'], b, e, observed)
                    ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
                    s2 = lik.robust_scores(ratio2, b, e, observed)
                    arms = {'flat': s_flat, 'source_likelihood': s0,
                            'covariance_likelihood': s1, 'class_projection': s2}
                    natural = {arm: natural_policy(b, e, s, ranking[qpos], priority, frac)
                               for arm, s in arms.items()}
                    kk = min(int(v['selected'].sum()) for v in natural.values())
                    matched = {arm: fixed_count(b, e, s, ranking[qpos], priority, kk) for arm, s in arms.items()}
                    y_q = y_ids[qpos]
                    for arm, s in arms.items():
                        for mode, mask in (('natural', natural[arm]['selected']), ('matched', matched[arm])):
                            pred = np.where(mask, expert[qpos], base[qpos])
                            records.append(dict(arm=arm, mode=mode, ba=macro_ba(y_q, pred)))
        rec = pd.DataFrame(records)
        per = rec.groupby(['arm', 'mode']).ba.agg(['mean', 'std']).reset_index()
        bf = per[(per['arm'] == 'flat') & (per['mode'] == 'matched')]['mean'].iloc[0]
        v = per[(per['arm'] == 'class_projection') & (per['mode'] == 'matched')]['mean'].iloc[0]
        nat = float(per[(per['arm'] == 'class_projection') & (per['mode'] == 'natural')]['mean'].iloc[0]
                    - per[(per['arm'] == 'flat') & (per['mode'] == 'natural')]['mean'].iloc[0])
        out['ports'][port] = {'n': int(d['n']), 'cp_flat': v - bf, 'natural': nat}
        print(f'{param}={value} {port}: {100*(v-bf):+.3f}pp', flush=True)
        if smoke:
            break
    plist = [p for p in out['ports'] if 'cp_flat' in out['ports'][p]]
    cp = np.array([out['ports'][p]['cp_flat'] for p in plist])
    nat = np.array([out['ports'][p]['natural'] for p in plist])
    res = {'param': param, 'value': value, 'n_ports': len(plist),
           'cp_flat_mean': float(cp.mean()), 'cp_flat_positive': int((cp > 0).sum()),
           'cp_flat_worst': float(cp.min()), 'natural_mean': float(nat.mean())}
    (OUT / f'sweep_{param}_{value}.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    (OUT / f'summary_{param}_{value}.json').write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))


def src_weights(labels, ports):
    result = np.empty(len(labels), dtype=np.float64)
    for port in np.unique(ports):
        mask = ports == port
        classes, inverse, counts = np.unique(labels[mask], return_inverse=True, return_counts=True)
        result[mask] = 1. / (len(np.unique(ports)) * len(classes) * counts[inverse])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--param', required=True); ap.add_argument('--value', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    build_cache()
    run_config(a.param, a.value, smoke=a.smoke)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()

