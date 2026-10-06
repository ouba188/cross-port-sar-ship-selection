"""模型池预算分配（MODELPOOL_PREREG_v1）：三模型池 + 源侧风险序分配 + oracle 天花板对照。"""
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

from bdh_pixel_loss_controls.product_correlated_source import fixed_count
from run_adaptive import CACHE, macro_ba, src_weights, SEEDS, ORDINALS, DRAWS, K

CACHE2 = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache2')
OUT = Path(r'E:/Hermes/paper_tracking/sensitivity/novel_arms/pool')
OUT.mkdir(parents=True, exist_ok=True)
EPSILON = 1e-12


def mlp_preds(Xtr, ytr, Xall):
    d = Xtr.shape[1]
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); Xall_t = torch.tensor(Xall, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    counts = np.bincount(ytr, minlength=K).astype(np.float64)
    w = (counts.sum() / (K * counts)).astype(np.float32)
    m = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, K))
    torch.manual_seed(0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w))
    for _ in range(30):
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), 128):
            idx = perm[i:i + 128]
            loss = crit(m(Xtr_t[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(Xall_t).numpy().argmax(1)


def run(smoke=False):
    out = {'ports': {}}
    for port in sorted(p.name[:-4] for p in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{port}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        d2 = np.load(CACHE2 / f'{port}.npz')
        Xs, Z = d2['Xs'], d2['Z']
        y_ids = d['y_ids']; qpos = d['qpos']; spos = d['spos']
        ports_all = d['ports_all']; ranking = d['ranking']
        base = d['base_logits'].argmax(1)
        ridge_e = d['expert_logits'].argmax(1)
        gbm = lgb.LGBMClassifier(n_estimators=150, num_leaves=4, learning_rate=0.1,
                                 n_jobs=1, verbose=-1, class_weight='balanced').fit(
            np.c_[Xs[spos], Z[spos]], y_ids[spos]).predict(np.c_[Xs, Z])
        mlp = mlp_preds(np.c_[Xs[spos], Z[spos]], y_ids[spos], np.c_[Xs, Z])
        pool = {'ridge': ridge_e, 'gbm': gbm, 'mlp': mlp}
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
            w = src_weights(y_ids[source_pos], ports_all[source_pos])
            # 源条件表 P(m 对 | y, base_pred)（对每个池成员）
            pair_probs = {}
            for name, mp in pool.items():
                num = np.bincount(K * y_ids[source_pos] + base[source_pos], weights=w * (mp[source_pos] == y_ids[source_pos]),
                                  minlength=K * K).reshape(K, K)
                den = np.bincount(K * y_ids[source_pos] + base[source_pos], weights=w, minlength=K * K).reshape(K, K)
                pair_probs[name] = np.divide(num, den, out=np.full_like(num, 1.0 / K), where=den > 0)
            for seed in SEEDS:
                for draw in range(DRAWS):
                    rng = np.random.default_rng(seed * 1000 + ordinal * 100 + draw)
                    ann = rng.choice(qpos, 64, replace=False)
                    ay = y_ids[ann]
                    observed = np.bincount(ay, minlength=K) > 0
                    priority = rng.permutation(query_n)
                    b, y_q = base[qpos], y_ids[qpos]
                    seen = np.flatnonzero(observed)          # 只用 observed 类（无目标标签泄漏）
                    num_b = np.bincount(K * y_ids[source_pos] + base[source_pos],
                                        weights=w * (base[source_pos] == y_ids[source_pos]), minlength=K * K).reshape(K, K)
                    den_b = np.bincount(K * y_ids[source_pos] + base[source_pos], weights=w, minlength=K * K).reshape(K, K)
                    p_base = np.divide(num_b, den_b, out=np.full_like(num_b, 1.0 / K), where=den_b > 0)
                    base_row = p_base[seen][:, b].mean(0)     # 每行 base 的期望正确率（observed 上平均）
                    gains = {}
                    for name, mp in pool.items():
                        gains[name] = pair_probs[name][seen][:, b].mean(0) - base_row
                    names = list(pool)
                    gmat = np.stack([gains[n] for n in names], axis=1)
                    best_idx = gmat.argmax(1)
                    u_star = gmat[np.arange(query_n), best_idx]
                    preds_matrix = np.stack([pool[n][qpos] for n in names], axis=1)
                    pred_star = preds_matrix[np.arange(query_n), best_idx]
                    eligible = np.flatnonzero((u_star > EPSILON) & (pred_star != b))
                    order = eligible[np.lexsort((priority[eligible], -ranking[qpos][eligible], -u_star[eligible]))]
                    k = int(np.clip(round(0.05 * query_n), 1, len(order))) if len(order) else 0
                    mask = np.zeros(query_n, dtype=bool)
                    if k:
                        mask[order[:k]] = True
                    pred = np.where(mask, pred_star, b)
                    # oracle 池（天花板，非部署）：逐样本取对的那个模型
                    correct = np.stack([pool[n][qpos] == y_q for n in names], axis=1)
                    oracle_pred = preds_matrix[np.arange(query_n), correct.argmax(1)]
                    records.append(dict(arm='pool_assign', ba=macro_ba(y_q, pred)))
                    records.append(dict(arm='oracle_pool', ba=macro_ba(y_q, oracle_pred)))
                    records.append(dict(arm='ridge_base', ba=macro_ba(y_q, b)))
        rec = pd.DataFrame(records)
        per = rec.groupby('arm').ba.mean()
        out['ports'][port] = {k: float(per[k]) for k in per.index}
        print(f'{port}: pool {per["pool_assign"]:.4f} oracle {per["oracle_pool"]:.4f} base {per["ridge_base"]:.4f}',
              flush=True)
        if smoke:
            break
    plist = list(out['ports'])
    pool_ba = np.array([out['ports'][p]['pool_assign'] for p in plist])
    base_ba = np.array([out['ports'][p]['ridge_base'] for p in plist])
    oracle_ba = np.array([out['ports'][p]['oracle_pool'] for p in plist])
    delta = pool_ba - base_ba
    res = {'mode': 'pool_assign', 'n_ports': len(plist), 'cp_flat_mean': float(delta.mean()),
           'cp_flat_positive': int((delta > 0).sum()), 'cp_flat_worst': float(delta.min()),
           'oracle_ba_mean': float(oracle_ba.mean()),
           'oracle_minus_base_mean': float((oracle_ba - base_ba).mean()),
           'oracle_capture': float(delta.mean() / (oracle_ba - base_ba).mean()) if (oracle_ba - base_ba).mean() > 0 else 0.0}
    (OUT / 'summary_pool.json').write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
