"""基线对比 BATCH 1（BASELINE_COMPARE_PREREG_v1）。8 臂同协议，冻结缓存复用。
用法：python run_baselines.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.linear_model import Ridge

from bdh_pixel_loss_controls.conditional_likelihood import fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import project_class_evidence_rt, K, natural_policy

OUT = Path(r'E:/Hermes/paper_tracking/sensitivity')
CACHE = OUT / 'pred_cache'
CACHE2 = OUT / 'pred_cache2'
ANNOT = 64
BUDGET = 0.05


def macro_ba(y, pred):
    present = np.unique(y)
    return float(np.mean([(pred[y == c] == c).mean() for c in present]))


def wacc(y, pred):
    return float((np.asarray(y) == np.asarray(pred)).mean())


def ridge_pred(tr_x, tr_y, te_x, alpha=1.0, n_cls=K):
    # balanced 用样本权重（sklearn Ridge 无 class_weight）：w=1/(n_cls·n_c)；多输出 one-hot→argmax
    tr_y = np.asarray(tr_y)
    _, inv, cnt = np.unique(tr_y, return_inverse=True, return_counts=True)
    sw = 1.0 / (len(cnt) * cnt[inv])
    Y = np.eye(n_cls)[tr_y]
    m = Ridge(alpha=alpha).fit(tr_x, Y, sample_weight=sw)
    return m.predict(te_x).argmax(1)


def _mat_sqrt(A):
    w, V = np.linalg.eigh(A)
    return (V * np.sqrt(np.clip(w, 0, None))[None]) @ V.T


def _mat_sqrt_inv(A):
    w, V = np.linalg.eigh(A)
    return (V * (1.0 / np.sqrt(np.clip(w, 1e-12, None)))[None]) @ V.T


def coral_align(src_x, tgt_x):
    """源二阶对齐到目标协方差（Sun et al. CORAL）。"""
    d = src_x.shape[1]
    Cs = np.cov(src_x, rowvar=False) + 1e-6 * np.eye(d)
    Ct = np.cov(tgt_x, rowvar=False) + 1e-6 * np.eye(d)
    return src_x @ _mat_sqrt_inv(Cs) @ _mat_sqrt(Ct)


def mmd_linear_map(src_x, tgt_x, steps=250, lr=2e-2, sigma=None, seed=0):
    """线性映射 W 最小化 RBF-MMD(源,目标)。返回 torch W(d,d)。"""
    import torch
    torch.manual_seed(seed)
    S = torch.tensor(src_x, dtype=torch.float32)
    T = torch.tensor(tgt_x, dtype=torch.float32)
    d = S.shape[1]
    # 子采样到最多 2000 行，控制 RBF 成对开销
    if S.shape[0] > 2000:
        idx = torch.randperm(S.shape[0])[:2000]
        S = S[idx]
    if T.shape[0] > 2000:
        idx = torch.randperm(T.shape[0])[:2000]
        T = T[idx]
    W = torch.eye(d, requires_grad=True)
    if sigma is None:
        with torch.no_grad():
            pool = torch.cat([S, T], 0)
            sigma = float(torch.pdist(pool).median()) + 1e-6
    opt = torch.optim.Adam([W], lr=lr)

    def mmd2(a, b):
        def k(x, y):
            xx = (x * x).sum(1, keepdim=True)
            yy = (y * y).sum(1, keepdim=True)
            d2 = xx + yy.T - 2 * (x @ y.T)
            return torch.exp(-d2 / (2 * sigma * sigma))
        kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
        return kaa.mean() + kbb.mean() - 2 * kab.mean()

    for _ in range(steps):
        opt.zero_grad()
        loss = mmd2(S @ W, T @ W)
        loss.backward()
        opt.step()
    return W.detach()


def run(smoke=False):
    rows = []
    for p in sorted(x.name[:-4] for x in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        d2 = np.load(CACHE2 / f'{p}.npz', allow_pickle=False)
        Xs, Z = d2['Xs'], d2['Z']
        feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
        qpos = d['qpos']; spos = d['spos']
        base = d['base_logits'].argmax(1); expert = d['expert_logits'].argmax(1)
        query_n = len(qpos); source_n = min(query_n, len(spos))
        rng_s = np.random.default_rng(1000)
        strat = {c: rng_s.choice(spos[y_ids[spos] == c],
                                 max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False)
                 for c in np.unique(y_ids[spos])}
        source_pos = np.concatenate(list(strat.values()))
        yq = y_ids[qpos]; bq = base[qpos]; eq = expert[qpos]
        X = np.hstack([Xs, Z])
        Xs_src, Xs_q = X[source_pos], X[qpos]

        # 1 VV-only
        vv = ridge_pred(Xs[source_pos], y_ids[source_pos], Xs[qpos])
        # 2 VH-only
        vh = ridge_pred(Z[source_pos], y_ids[source_pos], Z[qpos])
        # 3 Fusion = ERM([VV|VH]) = expert
        fu = eq
        # 4 CORAL
        coral_src = coral_align(Xs_src, Xs_q)
        co = ridge_pred(coral_src, y_ids[source_pos], Xs_q)
        # 5 MMD（线性映射）
        Wm = mmd_linear_map(Xs_src, Xs_q)
        mm = ridge_pred(Xs_src @ Wm.numpy(), y_ids[source_pos], Xs_q @ Wm.numpy())
        # 6/7 64 标签少样本（proto 只对观察到的类建原型；评估统一排除 ann 支持集）
        ann = np.random.default_rng(0).choice(qpos, ANNOT, replace=False)
        ay = y_ids[ann]
        observed_cls = np.unique(ay)
        means = np.stack([X[ann][ay == c].mean(0) for c in observed_cls])
        dist = ((X[qpos][:, None, :] - means[None]) ** 2).sum(2)
        pr = observed_cls[dist.argmin(1)]
        # ridge on 64
        rr = ridge_pred(X[ann], ay, X[qpos], n_cls=K)

        # 8 OTH（冻结链 natural 5%）
        source_model = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
        observed = np.bincount(ay, minlength=K) > 0
        p1 = shared_cov_predict(source_model, feats[ann], ay, feats[qpos])
        f1 = class_mass(p1['loglikelihood'], p1['counts'])
        ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
        s2 = robust_scores(ratio2, bq, eq, observed)
        priority = np.random.default_rng(2).permutation(query_n)
        nat = natural_policy(bq, eq, s2, d['ranking'][qpos], priority, BUDGET)
        ot = np.where(nat['selected'], eq, bq)

        arms = {'VV-only': vv, 'VH-only': vh, 'Fusion': fu, 'CORAL': co, 'MMD': mm,
                '64-proto': pr, '64-ridge': rr, 'OTH': ot}
        # 评估集 = query 排除 64 支持（少样本惯例，去除标注泄漏）
        in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
        ev = ~in_ann
        row = {'port': p, 'n': int(d['n'])}
        for a, pred in arms.items():
            row[f'ba_{a}'] = macro_ba(yq[ev], pred[ev])
            row[f'acc_{a}'] = wacc(yq[ev], pred[ev])
        # 成本效率：capture = (OTH-base)/(Fusion-base)，Fusion=100% 专家，OTH=5% 专家
        row['capture'] = ((row['ba_OTH'] - row['ba_VV-only']) /
                          max(row['ba_Fusion'] - row['ba_VV-only'], 1e-9))
        rows.append(row)
        ba = {a: f'{row[f"ba_{a}"]:.3f}' for a in arms}
        print(f'{p}: ' + ' '.join(f'{a}={ba[a]}' for a in arms) + f'  cap={row["capture"]:.2f}', flush=True)
        if smoke:
            break

    (OUT / 'baseline_compare.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    arms = ['VV-only', 'VH-only', 'Fusion', 'CORAL', 'MMD', '64-proto', '64-ridge', 'OTH']
    for a in arms:
        ba = np.array([r[f'ba_{a}'] for r in rows])
        acc = np.array([r[f'acc_{a}'] for r in rows])
        print(f'{a:9s} BA={ba.mean():.4f} (pos {(ba > 0).sum()}/{len(ba)})  ACC={acc.mean():.4f}')
    cap = np.array([r['capture'] for r in rows])
    print(f'capture=(OTH-VV)/(Fusion-VV): mean={cap.mean():.3f}  '
          f'(OTH 以 5% 专家成本取回 Fusion-VV 增益的 {100*cap.mean():.0f}%)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
