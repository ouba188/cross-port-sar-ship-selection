"""诊断 P1：ALS 随机初始化 vs 零初始化，看 rank-2 结构是否真实存在（而非被零初始化压塌）。
对比 rank-1 / rank-2 的重构误差。"""
import sys
from pathlib import Path
sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline'); sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')
import numpy as np
from bdh_pixel_loss_controls.conditional_likelihood import fit_source

CACHE = Path('pred_cache')


def fit_als(delta_cp, r, iters=50, random_init=True, seed=0):
    classes = sorted({c for (c, p) in delta_cp}); ports = sorted({p for (c, p) in delta_cp})
    d = next(iter(delta_cp.values())).shape[0]
    rng = np.random.default_rng(seed)
    f_p = {}
    for p in ports:
        if random_init:
            f_p[p] = rng.normal(0, 0.1, r)
        else:
            ds = [delta_cp[(c, p)] for c in classes if (c, p) in delta_cp]
            fp = np.zeros(r)
            if ds:
                fp[0] = float(np.linalg.norm(np.mean(ds, 0)))
            f_p[p] = fp
    W = {c: np.zeros((d, r)) for c in classes}
    for _ in range(iters):
        for c in classes:
            ps = [p for p in ports if (c, p) in delta_cp]
            if len(ps) < r + 1: continue
            F = np.stack([f_p[p] for p in ps]); D = np.stack([delta_cp[(c, p)] for p in ps])
            W[c] = np.linalg.lstsq(F, D, rcond=None)[0].T
        for p in ports:
            cs = [c for c in classes if (c, p) in delta_cp and np.linalg.norm(W[c]) > 1e-9]
            if not cs: continue
            A = sum(W[c].T @ W[c] for c in cs); b = sum(W[c].T @ delta_cp[(c, p)] for c in cs)
            f_p[p] = np.linalg.solve(A + 1e-6 * np.eye(r), b)
    # 重构误差
    err = 0.0; n = 0
    for (c, p), v in delta_cp.items():
        rec = W[c] @ f_p[p]
        err += float(np.sum((v - rec) ** 2)); n += 1
    # f_p[1] 的能量占比
    fp = np.stack([f_p[p] for p in ports])
    dim2_energy = float((fp[:, 1:] ** 2).sum() / (fp ** 2).sum()) if r >= 2 else 0.0
    return err / n, dim2_energy, f_p


for p in ['Antwerp-Bruges', 'Busan', 'Rotterdam']:
    d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
    if 'skipped' in d.files: continue
    feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']; qpos = d['qpos']; spos = d['spos']
    qn = len(qpos); sn = min(qn, len(spos))
    rng = np.random.default_rng(1000)
    strat = {c: rng.choice(spos[y_ids[spos] == c], max(1, int(round(sn * (y_ids[spos] == c).sum() / len(spos)))), replace=False) for c in np.unique(y_ids[spos])}
    sp = np.concatenate(list(strat.values()))
    sm = fit_source(feats[sp], y_ids[sp], ports_all[sp])
    z = (feats[sp] - sm['center']) / sm['scale']
    delta_cp = {}
    for c in np.unique(y_ids[sp]):
        for pp in np.unique(ports_all[sp]):
            m = (y_ids[sp] == c) & (ports_all[sp] == pp)
            if m.sum() >= 2:
                delta_cp[(c, pp)] = z[m].mean(0) - sm['means'][c]
    e1, d1, _ = fit_als(delta_cp, 1, random_init=False)
    e2_zero, d2z, _ = fit_als(delta_cp, 2, random_init=False)
    e2_rand, d2r, _ = fit_als(delta_cp, 2, random_init=True, seed=1)
    e2_rand2, d2r2, _ = fit_als(delta_cp, 2, random_init=True, seed=2)
    print(f'{p}: rank1 err={e1:.4f} | rank2(零init) err={e2_zero:.4f} dim2能量={d2z:.3f} | '
          f'rank2(随机init) err={min(e2_rand,e2_rand2):.4f} dim2能量={max(d2r,d2r2):.3f}')
