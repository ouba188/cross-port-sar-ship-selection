"""因果因子挖掘（跨港不变性）· CAUSAL_FACTOR_MINING_PREREG_v1。
用法：python run_causal_mining.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.linear_model import Ridge

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
CACHE2 = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache2')
K = 8
KS = [48, 96, 144]


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def wacc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def ridge_pred(tr_x, tr_y, te_x, alpha=1.0):
    tr_y = np.asarray(tr_y)
    _, inv, cnt = np.unique(tr_y, return_inverse=True, return_counts=True)
    sw = 1.0 / (len(cnt) * cnt[inv])
    m = Ridge(alpha=alpha).fit(tr_x, np.eye(K)[tr_y], sample_weight=sw)
    return m.predict(te_x).argmax(1)


def invariance_scores(X_s, y_s, p_s):
    """逐维不变性分数 I_j（小=不变=因果）。"""
    n, d = X_s.shape
    ports = np.unique(p_s)
    classes = np.unique(y_s)
    sig = X_s.std(0) + 1e-6
    score = np.zeros(d)
    for j in range(d):
        var_sum = 0.0; ncls = 0
        for c in classes:
            mus = []
            for q in ports:
                m = (y_s == c) & (p_s == q)
                if m.sum() >= 2:
                    mus.append(X_s[m, j].mean())
            if len(mus) >= 2:
                var_sum += float(np.var(mus))
                ncls += 1
        score[j] = (var_sum / max(ncls, 1)) / (sig[j] ** 2)
    return score


def run(smoke=False):
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
        X = np.hstack([Xs_f, Z_f])
        X_s, X_q = X[source_pos], X[qpos]
        y_s, y_q = y_ids[source_pos], y_ids[qpos]
        p_s = ports_all[source_pos]

        ann = np.random.default_rng(0).choice(qpos, 64, replace=False)
        in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
        ev = ~in_ann

        I = invariance_scores(X_s, y_s, p_s)
        order = np.argsort(I)  # 升序：小=不变=因果
        d = X_s.shape[1]

        row = {'port': p}
        # 参照：全维(fusion) 与 VV-only
        row['ba_full'] = macro_ba(y_q[ev], ridge_pred(X_s, y_s, X_q)[ev])
        row['ba_vv'] = macro_ba(y_q[ev], ridge_pred(X_s[:, :128], y_s, X_q[:, :128])[ev])
        for k in KS:
            inv_idx = order[:k]
            spu_idx = order[-k:]
            row[f'ba_inv{k}'] = macro_ba(y_q[ev], ridge_pred(X_s[:, inv_idx], y_s, X_q[:, inv_idx])[ev])
            row[f'ba_spu{k}'] = macro_ba(y_q[ev], ridge_pred(X_s[:, spu_idx], y_s, X_q[:, spu_idx])[ev])
        rows.append(row)
        print(f'{p}: full={row["ba_full"]:.3f} vv={row["ba_vv"]:.3f} ' +
              ' '.join(f'inv{k}={row[f"ba_inv{k}"]:.3f}/spu{k}={row[f"ba_spu{k}"]:.3f}"' for k in KS), flush=True)
        if smoke:
            break

    json.dump(rows, open('causal_mining.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('full   =', np.mean([r['ba_full'] for r in rows]))
    print('vv     =', np.mean([r['ba_vv'] for r in rows]))
    for k in KS:
        inv = np.mean([r[f'ba_inv{k}'] for r in rows]); spu = np.mean([r[f'ba_spu{k}'] for r in rows])
        print(f'inv{k} = {inv:.4f}   spu{k} = {spu:.4f}   Δ={inv-spu:+.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
