"""诊断：64-ridge 为什么 ACC 超我们——按类频分多数/少数，看每类准确率差。"""
import sys
from pathlib import Path
sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')
import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from bdh_pixel_loss_controls.conditional_likelihood import fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import project_class_evidence_rt, K

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
CACHE2 = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache2')

# 按类聚合：每类 (count, acc_64ridge, acc_ours, acc_fusion)
per_class = []
for p in sorted(x.name[:-4] for x in CACHE.glob('*.npz')):
    d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
    if 'skipped' in d.files: continue
    d2 = np.load(CACHE2 / f'{p}.npz', allow_pickle=False)
    Xs, Z = d2['Xs'], d2['Z']
    feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
    qpos = d['qpos']; spos = d['spos']
    base = d['base_logits'].argmax(1); expert = d['expert_logits'].argmax(1)
    query_n = len(qpos); source_n = min(query_n, len(spos))
    rng_s = np.random.default_rng(1000)
    strat = {c: rng_s.choice(spos[y_ids[spos] == c], max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False) for c in np.unique(y_ids[spos])}
    source_pos = np.concatenate(list(strat.values()))
    X = np.hstack([Xs, Z])
    yq = y_ids[qpos]; bq = base[qpos]; eq = expert[qpos]
    ann = np.random.default_rng(0).choice(qpos, 64, replace=False); ay = y_ids[ann]
    _, inv, cnt = np.unique(ay, return_inverse=True, return_counts=True)
    sw = 1.0 / (len(cnt) * cnt[inv])
    fs = Ridge(alpha=1.0).fit(X[ann], np.eye(K)[ay], sample_weight=sw).predict(X[qpos]).argmax(1)
    # 双向 s2
    observed = np.bincount(ay, minlength=K) > 0
    sm = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
    p1 = shared_cov_predict(sm, feats[ann], ay, feats[qpos])
    f1 = class_mass(p1['loglikelihood'], p1['counts'])
    ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
    s2 = robust_scores(ratio2, bq, eq, observed)
    D = (bq != eq)
    ours = np.where(D, np.where(s2 > 0, eq, bq), bq)
    in_ann = np.zeros(query_n, bool); in_ann[np.isin(qpos, ann)] = True
    ev = ~in_ann
    for c in np.unique(yq[ev]):
        m = (yq == c) & ev
        per_class.append((int(m.sum()), float((fs[m] == c).mean()), float((ours[m] == c).mean()), float((eq[m] == c).mean())))

arr = np.array(per_class, dtype=float)
# 按类样本数排序，分成两半：少数类 vs 多数类
order = np.argsort(arr[:, 0])
half = len(arr) // 2
minor = arr[order[:half]]; major = arr[order[half:]]
print(f'共 {len(arr)} 个(港×类)实例')
print(f'少数类(样本数中位 {minor[:,0].mean():.0f}): 64ridge={minor[:,1].mean():.3f}  ours={minor[:,2].mean():.3f}  fusion={minor[:,3].mean():.3f}')
print(f'多数类(样本数中位 {major[:,0].mean():.0f}): 64ridge={major[:,1].mean():.3f}  ours={major[:,2].mean():.3f}  fusion={major[:,3].mean():.3f}')
print(f'Δ(64ridge−ours): 少数类 { (minor[:,1]-minor[:,2]).mean():+.3f}   多数类 {(major[:,1]-major[:,2]).mean():+.3f}')
# 相关性：类样本数 vs (64ridge−ours)
from scipy.stats import spearmanr
print('Spearman(类样本数, 64ridge−ours) =', spearmanr(arr[:,0], arr[:,1]-arr[:,2]))
