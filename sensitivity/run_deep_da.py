"""深度对齐基线 DANN/IRM/MixStyle（DEEPN_DA_PREREG_v1）。torch 训练循环，冻结特征上。
用法：python run_deep_da.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits

import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
CACHE2 = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache2')
ANNOT = 64
K = 8
SEED = 0


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def wacc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grl(x, lam=1.0):
    return _GRL.apply(x, lam)


def _fit_linear(X_s, y_s, X_t, epochs, device, aug_fn=None):
    """线性分类器（可加 MixStyle 增广）；返回对 X_t 的 argmax。"""
    n, d = X_s.shape
    torch.manual_seed(SEED)
    w = nn.Linear(d, K).to(device)
    opt = torch.optim.Adam(w.parameters(), lr=1e-3)
    Xs = torch.tensor(X_s, dtype=torch.float32, device=device)
    ys = torch.tensor(y_s, dtype=torch.long, device=device)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            xb = Xs[idx]
            if aug_fn is not None:
                xb = aug_fn(xb)
            loss = F.cross_entropy(w(xb), ys[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    w.eval()
    with torch.no_grad():
        return w(torch.tensor(X_t, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()


def run_dann(X_s, y_s, X_t, device, epochs=200):
    n, d = X_s.shape
    torch.manual_seed(SEED)
    G = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 128)).to(device)
    C = nn.Linear(128, K).to(device)
    D = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
    opt = torch.optim.Adam(list(G.parameters()) + list(C.parameters()) + list(D.parameters()), lr=1e-3)
    Xs = torch.tensor(X_s, dtype=torch.float32, device=device)
    ys = torch.tensor(y_s, dtype=torch.long, device=device)
    Xt = torch.tensor(X_t, dtype=torch.float32, device=device)
    nt = Xt.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            xb = Xs[idx]
            f = G(xb)
            loss_cls = F.cross_entropy(C(f), ys[idx])
            # 域判别：源 vs 目标（子采样目标，数量 min(len(idx), nt)）
            k = min(len(idx), nt)
            ti = torch.randperm(nt, device=device)[:k]
            f_all = torch.cat([f, grl(G(Xt[ti]))], 0)
            dlab = torch.cat([torch.zeros(len(idx)), torch.ones(k)], 0).long().to(device)
            loss_dom = F.cross_entropy(D(f_all), dlab)
            loss = loss_cls + loss_dom
            opt.zero_grad(); loss.backward(); opt.step()
    G.eval(); C.eval()
    with torch.no_grad():
        return C(G(torch.tensor(X_t, dtype=torch.float32, device=device))).argmax(1).cpu().numpy()


def run_irm(X_s, y_s, envs, X_t, device, epochs=250, lam=1.0):
    """IRMv1：环境=源港。"""
    n, d = X_s.shape
    torch.manual_seed(SEED)
    w = nn.Linear(d, K).to(device)
    opt = torch.optim.Adam(w.parameters(), lr=1e-3)
    Xs = torch.tensor(X_s, dtype=torch.float32, device=device)
    ys = torch.tensor(y_s, dtype=torch.long, device=device)
    env_ids = np.unique(envs)
    for ep in range(epochs):
        opt.zero_grad()
        total = 0.0; penalty = 0.0
        for e in env_ids:
            m = torch.tensor(envs == e, device=device)
            if m.sum() < 2:
                continue
            xe = Xs[m]; ye = ys[m]
            dummy = torch.ones(1, device=device, requires_grad=True)
            logits = w(xe) * dummy
            le = F.cross_entropy(logits, ye)
            total = total + le
            ge = torch.autograd.grad(le, dummy, create_graph=True)[0]
            penalty = penalty + (ge ** 2).sum()
        ne = max(1, len(env_ids))
        loss = total / ne + lam * penalty / ne
        loss.backward(); opt.step()
    w.eval()
    with torch.no_grad():
        return w(torch.tensor(X_t, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()


def run_mixstyle(X_s, y_s, X_t, device, epochs=200):
    envs_avail = y_s is not None  # 用源样本做跨港风格混合：简化——跨样本混合
    def aug(xb):
        # MixStyle 简化：实例统计(mean/std over feature dims)跨样本混合
        mu = xb.mean(1, keepdim=True); sg = xb.std(1, keepdim=True) + 1e-5
        xn = (xb - mu) / sg
        idx = torch.randperm(xb.shape[0], device=xb.device)
        a = torch.distributions.Beta(torch.tensor(0.1), torch.tensor(0.1)).sample((xb.shape[0], 1)).to(xb.device)
        mu_m = a * mu + (1 - a) * mu[idx]; sg_m = a * sg + (1 - a) * sg[idx]
        mask = (torch.rand(xb.shape[0], 1, device=xb.device) < 0.5)
        xo = sg_m * xn + mu_m
        return torch.where(mask, xo, xb)
    return _fit_linear(X_s, y_s, X_t, epochs, device, aug_fn=aug)


def run(smoke=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device =', device, flush=True)
    rows = []
    for p in sorted(x.name[:-4] for x in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        d2 = np.load(CACHE2 / f'{p}.npz', allow_pickle=False)
        Xs_f, Z_f = d2['Xs'], d2['Z']
        y_ids = d['y_ids']; ports_all = d['ports_all']; qpos = d['qpos']; spos = d['spos']
        query_n = len(qpos); source_n = min(query_n, len(spos))
        rng_s = np.random.default_rng(1000)
        strat = {c: rng_s.choice(spos[y_ids[spos] == c],
                                 max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False)
                 for c in np.unique(y_ids[spos])}
        source_pos = np.concatenate(list(strat.values()))
        X = np.hstack([Xs_f, Z_f]).astype(np.float32)
        X_s = X[source_pos]; X_t = X[qpos]
        y_s = y_ids[source_pos]; yq = y_ids[qpos]
        envs = ports_all[source_pos]

        y_dann = run_dann(X_s, y_s, X_t, device)
        y_irm = run_irm(X_s, y_s, envs, X_t, device)
        y_mix = run_mixstyle(X_s, y_s, X_t, device)

        ann = np.random.default_rng(0).choice(qpos, ANNOT, replace=False)
        in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
        ev = ~in_ann
        row = {'port': p, 'n': int(d['n'])}
        for name, pr in [('DANN', y_dann), ('IRM', y_irm), ('MixStyle', y_mix)]:
            row[f'ba_{name}'] = macro_ba(yq[ev], pr[ev])
            row[f'acc_{name}'] = wacc(yq[ev], pr[ev])
        rows.append(row)
        print(f'{p}: DANN={row["ba_DANN"]:.3f} IRM={row["ba_IRM"]:.3f} MixStyle={row["ba_MixStyle"]:.3f}', flush=True)
        if smoke:
            break

    json.dump(rows, open('deep_da.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for name in ['DANN', 'IRM', 'MixStyle']:
        ba = np.mean([r[f'ba_{name}'] for r in rows]); acc = np.mean([r[f'acc_{name}'] for r in rows])
        print(f'{name:9s} BA={ba:.4f} ACC={acc:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
