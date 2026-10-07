"""诊断：门控误差到底是不是"类均值不准"造成的？
比较三类均值的门控 BA：
1. 源均值（独立适配，当前）
2. 64 标签均值（独立适配）
3. oracle 均值（用全目标标签估 μ_T，完美均值的上界）
若 3 只比 1/2 高一点点 ⇒ 门控瓶颈不在均值，B 的方向性错误。
"""
import sys
from pathlib import Path
sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline'); sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')
import numpy as np
from threadpoolctl import threadpool_limits
from scipy.stats import multivariate_t
from bdh_pixel_loss_controls.conditional_likelihood import fit_source, TAU
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import K

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
SEEDS = [0, 1, 2, 3, 4]


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def gate_ll(ll, bq, eq):
    D = (bq != eq)
    dd = ll[np.arange(len(bq)), eq] - ll[np.arange(len(bq)), bq]
    return np.where(D, np.where(dd > 0, eq, bq), bq)


def ll_with_means(source, query_x, means):
    """给定均值，算 Student-t 似然（协方差/df 用 source 的）。"""
    query = (query_x - source['center']) / source['scale']
    d = source['means'].shape[1]
    psi = TAU * source['covariance']
    df = d + 1 + TAU + 0 - d + 1  # 无 64 标签时 df = TAU+2
    sf = (1 + 1 / TAU) / df
    return np.column_stack([multivariate_t.logpdf(query, loc=means[c], shape=psi * sf, df=df) for c in range(K)])


rows = []
for p in sorted(x.name[:-4] for x in CACHE.glob('*.npz')):
    d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
    if 'skipped' in d.files:
        continue
    feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']; qpos = d['qpos']; spos = d['spos']
    base = d['base_logits'].argmax(1); expert = d['expert_logits'].argmax(1)
    yq = y_ids[qpos]; bq = base[qpos]; eq = expert[qpos]
    query_n = len(qpos); source_n = min(query_n, len(spos))
    rng_s = np.random.default_rng(1000)
    strat = {c: rng_s.choice(spos[y_ids[spos] == c], max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False) for c in np.unique(y_ids[spos])}
    source_pos = np.concatenate(list(strat.values()))
    sm = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
    z_q = (feats[qpos] - sm['center']) / sm['scale']
    # oracle 均值（全目标标签）
    oracle_means = np.stack([z_q[yq == c].mean(0) for c in range(K)])
    ll_src = ll_with_means(sm, feats[qpos], sm['means'])
    ll_oracle = ll_with_means(sm, feats[qpos], oracle_means)
    g_src = gate_ll(ll_src, bq, eq)
    g_oracle = gate_ll(ll_oracle, bq, eq)
    # 64 标签版（独立适配，多 seed 均值）
    g_64 = []
    for seed in SEEDS:
        ann = np.random.default_rng(seed).choice(qpos, 64, replace=False)
        ay = y_ids[ann]
        in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
        ev = ~in_ann
        ll = shared_cov_predict(sm, feats[ann], ay, feats[qpos])['loglikelihood']
        g = gate_ll(ll, bq, eq)
        g_64.append(macro_ba(yq[ev], g[ev]))
    # 源/oracle 用同一 ev（seed0 的 ann）
    ann = np.random.default_rng(0).choice(qpos, 64, replace=False)
    in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
    ev = ~in_ann
    ba_src = macro_ba(yq[ev], g_src[ev]); ba_oracle = macro_ba(yq[ev], g_oracle[ev])
    rows.append((p, ba_src, np.mean(g_64), ba_oracle))
    print(f'{p}: 源均值={ba_src:.4f} 64标签均值={np.mean(g_64):.4f} oracle均值={ba_oracle:.4f} '
          f'(oracle-源=+{ba_oracle-ba_src:.4f})', flush=True)

print('\n=== 汇总 ===')
for i, name in [(1, '源均值'), (2, '64标签均值'), (3, 'oracle均值')]:
    v = np.array([r[i] for r in rows])
    print(f'{name}: BA={v.mean():.4f}')
print(f'oracle 均值相对源均值提升 = {np.array([r[3]-r[1] for r in rows]).mean():+.4f}')
