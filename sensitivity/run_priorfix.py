"""键的标签漂移校正（POOL_PRIORFIX_PREREG_v1）：BBSE/EM 估目标先验 → 重加权键表 → 5% 预算排序。
对照臂：base / rand5 / rank5（原键）/ rank5_pp（先验校正键）/ truegain5（上界）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
import pandas as pd
import lightgbm as lgb
from threadpoolctl import threadpool_limits

from run_adaptive import CACHE, macro_ba, src_weights, ORDINALS, K
from diagnose_modelpool import mlp_preds

CACHE2 = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache2')
OUT = Path(r'E:/Hermes/paper_tracking/sensitivity/novel_arms/pool')
PORTS = ['Antwerp-Bruges', 'Singapore', 'Qingdao', 'Rotterdam', 'Busan', 'Los Angeles', 'Santos', 'Mombasa']
BUDGET = 0.05


def bbse_prior(C, q, pi0, iters=20):
    """EM 标签漂移：C[i,c]=P(pred=i|y=c)；q[i]=目标边际。返回 π̂_tgt。"""
    pi = pi0.copy()
    for _ in range(iters):
        pred_marg = C @ pi
        pred_marg = np.maximum(pred_marg, 1e-12)
        pi = pi * (C.T @ (q / pred_marg))
        s = pi.sum()
        pi = pi / s if s > 0 else pi0.copy()
    return pi


def run():
    recs = []
    for port in PORTS:
        if not (CACHE / f'{port}.npz').exists():
            continue
        d = np.load(CACHE / f'{port}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        d2 = np.load(CACHE2 / f'{port}.npz')
        Xs, Z = d2['Xs'], d2['Z']
        y_ids, qpos, spos, ports_all, ranking = d['y_ids'], d['qpos'], d['spos'], d['ports_all'], d['ranking']
        base = d['base_logits'].argmax(1)
        rid = d['expert_logits'].argmax(1)
        gbm = lgb.LGBMClassifier(n_estimators=150, num_leaves=4, learning_rate=0.1, n_jobs=1,
                                 verbose=-1, class_weight='balanced').fit(
            np.c_[Xs[spos], Z[spos]], y_ids[spos]).predict(np.c_[Xs, Z])
        mlp = mlp_preds(np.c_[Xs[spos], Z[spos]], y_ids[spos], np.c_[Xs, Z])
        pool = {'ridge': rid, 'gbm': gbm, 'mlp': mlp}
        names = list(pool)
        query_n = len(qpos)
        source_n = min(query_n, len(spos))
        rows = []
        for ordinal in ORDINALS:
            rng_s = np.random.default_rng(1000 + ordinal)
            strat = {c: rng_s.choice(spos[y_ids[spos] == c],
                                     max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False)
                     for c in np.unique(y_ids[spos])}
            source_pos = np.concatenate(list(strat.values()))
            w = src_weights(y_ids[source_pos], ports_all[source_pos])
            # 源侧键表 + 混淆矩阵 + 源先验
            pp, C, pi_src = {}, np.zeros((K, K)), np.bincount(y_ids[source_pos], weights=w, minlength=K)
            pi_src = pi_src / max(pi_src.sum(), 1e-12)
            for n, mp in pool.items():
                num = np.bincount(K * y_ids[source_pos] + base[source_pos],
                                  weights=w * (mp[source_pos] == y_ids[source_pos]), minlength=K * K).reshape(K, K)  # [y, base]
                den = np.bincount(K * y_ids[source_pos] + base[source_pos], weights=w, minlength=K * K).reshape(K, K)
                pp[n] = np.divide(num, den, out=np.full_like(num, 1.0 / K), where=den > 0)
                if n == 'ridge':
                    C = den.T.copy()
                    C = C / np.maximum(C.sum(0, keepdims=True), 1e-12)   # [pred, y]
            # 目标边际（未标注行）
            q = np.bincount(base[qpos], minlength=K).astype(float); q /= max(q.sum(), 1e-12)
            pi_hat = bbse_prior(C, q, pi_src)
            ratio = np.maximum(pi_hat, 1e-9) / np.maximum(pi_src, 1e-9)
            for seed in (0, 1, 2, 3, 4):
                rng = np.random.default_rng(seed * 1000 + ordinal)
                ann = rng.choice(qpos, 64, replace=False)
                observed = np.bincount(y_ids[ann], minlength=K) > 0
                seen = np.flatnonzero(observed)
                prio = rng.permutation(query_n)
                b, yq = base[qpos], y_ids[qpos]
                pmat = np.stack([pool[n][qpos] for n in names], 1)
                correct = np.stack([pool[n][qpos] == yq for n in names], 1)
                base_ok = (b == yq)
                base_row = pp['ridge'][seen][:, b].mean(0)
                g_raw = np.stack([pp[n][seen][:, b].mean(0) - base_row for n in names], 1)
                # 先验校正键
                ppp = {n: (pp[n].T * ratio).T for n in names}          # 按类别轴重加权
                ppp = {n: v / np.maximum(v.sum(0, keepdims=True), 1e-12) for n, v in ppp.items()}
                base_row_pp = ppp['ridge'][seen][:, b].mean(0)
                g_pp = np.stack([ppp[n][seen][:, b].mean(0) - base_row_pp for n in names], 1)
                k = int(BUDGET * query_n)
                out = {}
                for tag, g in (('rank5', g_raw), ('rank5_pp', g_pp)):
                    best = g.argmax(1)
                    u = g[np.arange(query_n), best]
                    pstar = pmat[np.arange(query_n), best]
                    elig = np.flatnonzero((u > 1e-12) & (pstar != b))
                    kk = min(k, len(elig))
                    order = elig[np.lexsort((prio[elig], -ranking[qpos][elig], -u[elig]))][:kk] if kk else np.array([], int)
                    mask = np.zeros(query_n, bool); mask[order] = True
                    out[tag] = macro_ba(yq, np.where(mask, pstar, b))
                dis = np.flatnonzero(b != pmat[:, 0])
                trueg = np.where(base_ok[dis], 0, np.max(np.where(~base_ok[dis, None], correct[dis], False), axis=1))
                order_t = dis[np.lexsort((prio[dis], -ranking[qpos][dis], -trueg))][:min(k, len(dis))]
                mask = np.zeros(query_n, bool); mask[order_t] = True
                out['truegain5'] = macro_ba(yq, np.where(mask, pmat[np.arange(query_n), g_raw.argmax(1)], b))
                out['rand5'] = macro_ba(yq, np.where(
                    np.isin(np.arange(query_n), dis[rng.permutation(len(dis))[:min(k, len(dis))]]),
                    pmat[np.arange(query_n), g_raw.argmax(1)], b))
                out['base'] = macro_ba(yq, b)
                rows.append(out)
        m = pd.DataFrame(rows).mean()
        recs.append({'port': port, **{c: float(m[c]) for c in m.index}})
        print(port, {c: round(float(m[c]), 4) for c in m.index}, flush=True)
    res = pd.DataFrame(recs).mean(numeric_only=True).to_dict()
    res = {k: float(v) for k, v in res.items() if k != 'port'}
    d = {f'{a}_minus_base': res[a] - res['base'] for a in ('rand5', 'rank5', 'rank5_pp', 'truegain5')}
    d['rank5_pp_minus_rand5'] = res['rank5_pp'] - res['rand5']
    d['rank5_pp_minus_rank5'] = res['rank5_pp'] - res['rank5']
    head = res['truegain5'] - res['rand5']
    d['capture_vs_head'] = (res['rank5_pp'] - res['rand5']) / head if head > 0 else 0.0
    (OUT / 'priorfix_result.json').write_text(json.dumps({'abs': res, 'delta': d}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(d, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
