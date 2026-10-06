"""条件关系迁移（CONDMIGRATE_PREREG_v1）：用主线 M2/M3 的 ratio2 类证据做池模型选择。
臂：base / rand5 / rank5（朴素键）/ rank5_mc（主线信号）/ truegain5。
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

from bdh_pixel_loss_controls import conditional_likelihood as lik
from bdh_pixel_loss_controls import shared_covariance_likelihood as shared_cov
from run_adaptive import (CACHE, project_class_evidence_rt, macro_ba, src_weights,
                          ORDINALS, K)
from diagnose_modelpool import mlp_preds

CACHE2 = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache2')
OUT = Path(r'E:/Hermes/paper_tracking/sensitivity/novel_arms/pool')
PORTS = ['Antwerp-Bruges', 'Singapore', 'Qingdao', 'Rotterdam', 'Busan', 'Los Angeles', 'Santos', 'Mombasa']
BUDGET = 0.05
EPS = 1e-12


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
        feats = d['feats']
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
            source_model = lik.fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
            # 朴素键表（rank5 参照）
            pp = {}
            for n, mp in pool.items():
                num = np.bincount(K * y_ids[source_pos] + base[source_pos],
                                  weights=w * (mp[source_pos] == y_ids[source_pos]), minlength=K * K).reshape(K, K)
                den = np.bincount(K * y_ids[source_pos] + base[source_pos], weights=w, minlength=K * K).reshape(K, K)
                pp[n] = np.divide(num, den, out=np.full_like(num, 1.0 / K), where=den > 0)
            for seed in (0, 1, 2, 3, 4):
                rng = np.random.default_rng(seed * 1000 + ordinal)
                ann = rng.choice(qpos, 64, replace=False)
                ay = y_ids[ann]
                observed = np.bincount(ay, minlength=K) > 0
                seen = np.flatnonzero(observed)
                prio = rng.permutation(query_n)
                b, yq = base[qpos], y_ids[qpos]
                pmat = np.stack([pool[n][qpos] for n in names], 1)          # (nq, 3)
                correct = np.stack([pool[n][qpos] == yq for n in names], 1)
                base_ok = (b == yq)
                # 主线信号 ratio2
                p1 = shared_cov.predict(source_model, feats[ann], ay, feats[qpos])
                f1 = lik.class_mass(p1['loglikelihood'], p1['counts'])
                ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])   # (nq, K)
                # 主线信号：行 i 对模型 j 的一致度 = ratio2[i, pred_j[i]]
                agree = ratio2[np.arange(query_n)[:, None], pmat]          # (nq, 3)
                agree_base = ratio2[np.arange(query_n), b]
                k = int(BUDGET * query_n)
                out = {}
                # 朴素键 rank5
                base_row = pp['ridge'][seen][:, b].mean(0)
                g_raw = np.stack([pp[n][seen][:, b].mean(0) - base_row for n in names], 1)
                best = g_raw.argmax(1); u = g_raw[np.arange(query_n), best]; pstar = pmat[np.arange(query_n), best]
                elig = np.flatnonzero((u > EPS) & (pstar != b))
                kk = min(k, len(elig))
                order = elig[np.lexsort((prio[elig], -ranking[qpos][elig], -u[elig]))][:kk] if kk else np.array([], int)
                mask = np.zeros(query_n, bool); mask[order] = True
                out['rank5'] = macro_ba(yq, np.where(mask, pstar, b))
                # 主线信号 rank5_mc
                best_mc = agree.argmax(1)
                u_mc = agree[np.arange(query_n), best_mc] - agree_base
                pstar_mc = pmat[np.arange(query_n), best_mc]
                elig_mc = np.flatnonzero((u_mc > EPS) & (pstar_mc != b))
                kk = min(k, len(elig_mc))
                order = elig_mc[np.lexsort((prio[elig_mc], -ranking[qpos][elig_mc], -u_mc[elig_mc]))][:kk] if kk else np.array([], int)
                mask = np.zeros(query_n, bool); mask[order] = True
                out['rank5_mc'] = macro_ba(yq, np.where(mask, pstar_mc, b))
                # rand5 / truegain5（同 priorfix 口径）
                dis = np.flatnonzero(b != pmat[:, 0])
                trueg = np.where(base_ok[dis], 0, np.max(np.where(~base_ok[dis, None], correct[dis], False), axis=1))
                order_t = dis[np.lexsort((prio[dis], -ranking[qpos][dis], -trueg))][:min(k, len(dis))]
                mask = np.zeros(query_n, bool); mask[order_t] = True
                out['truegain5'] = macro_ba(yq, np.where(mask, pmat[np.arange(query_n), best_mc], b))
                mask = np.zeros(query_n, bool)
                mask[dis[rng.permutation(len(dis))[:min(k, len(dis))]]] = True
                out['rand5'] = macro_ba(yq, np.where(mask, pmat[np.arange(query_n), best_mc], b))
                out['base'] = macro_ba(yq, b)
                rows.append(out)
        m = pd.DataFrame(rows).mean()
        recs.append({'port': port, **{c: float(m[c]) for c in m.index}})
        print(port, {c: round(float(m[c]), 4) for c in m.index}, flush=True)
    res = pd.DataFrame(recs).mean(numeric_only=True).to_dict()
    res = {k: float(v) for k, v in res.items() if k != 'port'}
    d = {f'{a}_minus_base': res[a] - res['base'] for a in ('rand5', 'rank5', 'rank5_mc', 'truegain5')}
    d['rank5_mc_minus_rand5'] = res['rank5_mc'] - res['rand5']
    d['rank5_mc_minus_rank5'] = res['rank5_mc'] - res['rank5']
    head = res['truegain5'] - res['rand5']
    d['capture_vs_head'] = (res['rank5_mc'] - res['rand5']) / head if head > 0 else 0.0
    (OUT / 'condmigrate_result.json').write_text(json.dumps({'abs': res, 'delta': d}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(d, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
