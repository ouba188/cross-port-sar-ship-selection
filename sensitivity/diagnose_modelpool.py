"""模型池互补性诊断：ridge/GBM/MLP 三预测器逐样本互补率（池分配的前置门槛）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from threadpoolctl import threadpool_limits

from run_adaptive import CACHE, K

CACHE2 = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache2')
OUT = Path(r'E:/Hermes/paper_tracking/sensitivity/novel_arms')
RES = OUT / 'modelpool_complementarity.json'


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


def main():
    from sklearn.linear_model import RidgeClassifier
    per_port = {}
    for port in sorted(p.name[:-4] for p in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{port}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        d2 = np.load(CACHE2 / f'{port}.npz')
        Xs, Z = d2['Xs'], d2['Z']
        y_ids = d['y_ids']; qpos = d['qpos']; spos = d['spos']
        # ridge（冻结缓存里已有 logits）作为模型 A
        bl = d['base_logits']; el = d['expert_logits']
        ridge_pred = el[qpos].argmax(1)   # 专家 ridge 作为池成员之一
        gbm_pred = lgb.LGBMClassifier(n_estimators=150, num_leaves=4, learning_rate=0.1,
                                      n_jobs=1, verbose=-1, class_weight='balanced').fit(
            np.c_[Xs[spos], Z[spos]], y_ids[spos]).predict(np.c_[Xs[qpos], Z[qpos]])
        mlp_pred = mlp_preds(np.c_[Xs[spos], Z[spos]], y_ids[spos], np.c_[Xs[qpos], Z[qpos]])
        yq = y_ids[qpos]
        preds = {'ridge': ridge_pred, 'gbm': gbm_pred, 'mlp': mlp_pred}
        acc = {k: float((v == yq).mean()) for k, v in preds.items()}
        # 互补率：三模型中至少一个对，而"任选其一"未必对 → 用 pairwise 对错差
        names = list(preds)
        pair_comp = {}
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = names[i], names[j]
                # b 对而 a 错 的比例（b 相对 a 的互补）
                pair_comp[f'{a}->{b}'] = float(((preds[b] == yq) & (preds[a] != yq)).mean())
        # 池 oracle：三模型逐样本取最好
        pool_oracle = float(np.maximum.reduce([v == yq for v in preds.values()]).mean())
        per_port[port] = dict(acc=acc, pair_comp=pair_comp, pool_oracle=pool_oracle)
        print(port, json.dumps(dict(acc=acc, pool_oracle=round(pool_oracle, 4)), ensure_ascii=False))
    RES.write_text(json.dumps(per_port, ensure_ascii=False, indent=1), encoding='utf-8')
    # 汇总：互补上界 = pool_oracle − 单模型最优 acc（港级均值）
    best = np.mean([max(v['acc'].values()) for v in per_port.values()])
    pool = np.mean([v['pool_oracle'] for v in per_port.values()])
    summ = dict(best_single_model_acc=float(best), pool_oracle_acc=float(pool),
                pool_gain_pp=float(100 * (pool - best)))
    (OUT / 'modelpool_summary.json').write_text(json.dumps(summ, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
