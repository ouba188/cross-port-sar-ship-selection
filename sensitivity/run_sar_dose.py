"""SAR 原生剂量-响应（SAR_DOSE_RESPONSE_PREREG_v1）。α 衰减 VH 通道，测 H→G 单调性。
用法：python run_sar_dose.py [--smoke]
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

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
CACHE2 = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache2')
K = 8
ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def ridge_pred(tr_x, tr_y, te_x, alpha=1.0):
    tr_y = np.asarray(tr_y)
    _, inv, cnt = np.unique(tr_y, return_inverse=True, return_counts=True)
    sw = 1.0 / (len(cnt) * cnt[inv])
    m = Ridge(alpha=alpha).fit(tr_x, np.eye(K)[tr_y], sample_weight=sw)
    return m.predict(te_x).argmax(1)


def run(smoke=False):
    # 结构：per_port 存每档 α 的 (H, G_oracle)
    curve = {a: [] for a in ALPHAS}  # 每档存 (H, G) 对
    for p in sorted(x.name[:-4] for x in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        d2 = np.load(CACHE2 / f'{p}.npz', allow_pickle=False)
        Xs_f, Z_f = d2['Xs'], d2['Z']
        y_ids = d['y_ids']; qpos = d['qpos']; spos = d['spos']
        query_n = len(qpos); source_n = min(query_n, len(spos))
        rng_s = np.random.default_rng(1000)
        strat = {c: rng_s.choice(spos[y_ids[spos] == c],
                                 max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False)
                 for c in np.unique(y_ids[spos])}
        source_pos = np.concatenate(list(strat.values()))
        y_s, y_q = y_ids[source_pos], y_ids[qpos]
        Xs_s, Xs_q = Xs_f[source_pos], Xs_f[qpos]
        Z_s, Z_q = Z_f[source_pos], Z_f[qpos]

        ann = np.random.default_rng(0).choice(qpos, 64, replace=False)
        in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
        ev = ~in_ann

        base_q = ridge_pred(Xs_s, y_s, Xs_q)
        base_ba = macro_ba(y_q[ev], base_q[ev])
        bc = (base_q == y_q); 
        for a in ALPHAS:
            Xs_a = np.hstack([Xs_s, a * Z_s]); Xq_a = np.hstack([Xs_q, a * Z_q])
            ex_q = ridge_pred(Xs_a, y_s, Xq_a)
            H = macro_ba(y_q[ev], ex_q[ev]) - base_ba
            ec = (ex_q == y_q)
            D = (base_q != ex_q)
            # oracle 双向：分歧样本挑对的那个
            oracle = np.where(D, np.where(ec, ex_q, base_q), base_q)
            G = macro_ba(y_q[ev], oracle[ev]) - base_ba
            curve[a].append((H, G))
        if smoke:
            break

    json.dump({str(a): v for a, v in curve.items()}, open('sar_dose.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    from scipy.stats import spearmanr
    Hs = []; Gs = []
    for a in ALPHAS:
        Hs.append(np.mean([v[0] for v in curve[a]]))
        Gs.append(np.mean([v[1] for v in curve[a]]))
        print(f'α={a:.3f}  H={Hs[-1]:+.4f}  G_oracle={Gs[-1]:+.4f}')
    rho, pval = spearmanr(Hs, Gs)
    print(f'Spearman(H, G_oracle) = {rho:.4f}  p={pval:.4f}')
    print('判据: ', 'PASS' if (rho >= 0.8 and pval < 0.01) else 'FAIL')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
