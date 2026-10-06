"""模型池失败归因诊断：在 5% 绑定预算下加两个对照臂，分离「预算不够」vs「源侧信号无信息」。
臂：base / rand5（随机选分歧行）/ rank5（源侧风险序）/ truegain5（目标标签排序，非部署上界）。
若 truegain5 ≫ rand5 ⇒ 预算内有可拣的头部，失败在源侧信号；
若 truegain5 ≈ rand5 ⇒ 5% 预算内可换的行同质，失败在预算/余量。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from threadpoolctl import threadpool_limits

from run_adaptive import CACHE, macro_ba, src_weights, ORDINALS, K
from diagnose_modelpool import mlp_preds

CACHE2 = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache2')
OUT = Path(r'E:/Hermes/paper_tracking/sensitivity/novel_arms/pool')
PORTS = ['Antwerp-Bruges', 'Singapore', 'Qingdao', 'Rotterdam', 'Busan', 'Los Angeles', 'Santos', 'Mombasa']
BUDGET = 0.05


def run():
    rows = []
    ports = [p for p in PORTS if (CACHE / f'{p}.npz').exists()]
    print('端口', ports, flush=True)
    for port in ports:
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
        rec = []
        for ordinal in ORDINALS:
            rng_s = np.random.default_rng(1000 + ordinal)
            strat = {c: rng_s.choice(spos[y_ids[spos] == c],
                                     max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False)
                     for c in np.unique(y_ids[spos])}
            source_pos = np.concatenate(list(strat.values()))
            w = src_weights(y_ids[source_pos], ports_all[source_pos])
            pp = {}
            for n, mp in pool.items():
                num = np.bincount(K * y_ids[source_pos] + base[source_pos],
                                  weights=w * (mp[source_pos] == y_ids[source_pos]), minlength=K * K).reshape(K, K)
                den = np.bincount(K * y_ids[source_pos] + base[source_pos], weights=w, minlength=K * K).reshape(K, K)
                pp[n] = np.divide(num, den, out=np.full_like(num, 1.0 / K), where=den > 0)
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
                # 源侧风险序（与预注册同）
                base_row = pp['ridge'][seen][:, b].mean(0)
                gains = np.stack([pp[n][seen][:, b].mean(0) - base_row for n in names], 1)
                best = gains.argmax(1)
                u = gains[np.arange(query_n), best]
                pstar = pmat[np.arange(query_n), best]
                elig = np.flatnonzero((u > 1e-12) & (pstar != b))
                k = min(int(BUDGET * query_n), len(elig)) if len(elig) else 0
                # 目标标签真增益（非部署上界）：分歧行里"换了就变对"优先
                dis = np.flatnonzero(b != pmat[:, 0])   # ridge 与 base 分歧
                trueg = np.where(base_ok[dis], 0, np.max(np.where(~base_ok[dis, None],
                                correct[dis], False), axis=1))
                order_t = dis[np.lexsort((prio[dis], -ranking[qpos][dis], -trueg))]
                order_r = elig[np.lexsort((prio[elig], -ranking[qpos][elig], -u[elig]))] if len(elig) else np.array([], int)
                arms = {}
                for nm, idx in (('rand5', dis[rng.permutation(len(dis))[:min(k, len(dis))]]),
                                ('rank5', order_r[:k]),
                                ('truegain5', order_t[:min(k, len(dis))])):
                    mask = np.zeros(query_n, bool); mask[idx] = True
                    arms[nm] = macro_ba(yq, np.where(mask, pmat[np.arange(query_n), best], b))
                arms['base'] = macro_ba(yq, b)
                arms['expert_all'] = macro_ba(yq, pmat[:, 0])
                rec.append(arms)
        dfp = pd.DataFrame(rec).mean()
        rows.append({'port': port, **{k: float(v) for k, v in dfp.items()}})
        print(port, {k: round(float(v), 4) for k, v in dfp.items()}, flush=True)
    res = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    res = {k: float(v) for k, v in res.items() if k != 'port'}
    deltas = {f'{a}_minus_base': res[a] - res['base'] for a in ('rand5', 'rank5', 'truegain5')}
    (OUT / 'attribution_pool.json').write_text(json.dumps({'abs': res, 'delta': deltas}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(deltas, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
