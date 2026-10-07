"""验证 GPT 的 OT 拆解主张：熵正则 OT 下 log ρ_e - log ρ_b == ℓ_e - ℓ_b + β_e - β_b？
即：OT 对 base-vs-expert 比较的贡献 = 批次级类别偏置（与样本无关的常数）。"""
import sys
from pathlib import Path
sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')
import numpy as np
from threadpoolctl import threadpool_limits
from bdh_pixel_loss_controls.conditional_likelihood import fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import project_class_evidence_rt, K

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')

p = 'Antwerp-Bruges'
d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
qpos = d['qpos']; spos = d['spos']
base = d['base_logits'].argmax(1); expert = d['expert_logits'].argmax(1)
query_n = len(qpos); source_n = min(query_n, len(spos))
rng_s = np.random.default_rng(1000)
strat = {c: rng_s.choice(spos[y_ids[spos] == c], max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False) for c in np.unique(y_ids[spos])}
source_pos = np.concatenate(list(strat.values()))
ann = np.random.default_rng(0).choice(qpos, 64, replace=False); ay = y_ids[ann]
bq = base[qpos]; eq = expert[qpos]
observed = np.bincount(ay, minlength=K) > 0
sm = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
p1 = shared_cov_predict(sm, feats[ann], ay, feats[qpos])
f1 = class_mass(p1['loglikelihood'], p1['counts'])
ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])

# 分歧样本
D = (bq != eq)
loglik = p1['loglikelihood']
raw_diff = loglik[np.arange(query_n), eq] - loglik[np.arange(query_n), bq]      # ℓ_e - ℓ_b
logratio = np.log(ratio2[np.arange(query_n), eq] / ratio2[np.arange(query_n), bq])  # log ρ_e - log ρ_b
resid = logratio - raw_diff                                                    # 应为 β_e - β_b（仅依赖类对）

# 检验：resid 是否只随 (b,e) 类对变化、不随样本变化
pairs = K * bq + eq
for pr in np.unique(pairs[D]):
    m = D & (pairs == pr)
    vals = resid[m]
    print(f'类对(b={bq[m][0]},e={eq[m][0]}): n={m.sum()}  resid mean={vals.mean():.4f}  std={vals.std():.6f}')
print('\n若每类对 std≈0 ⇒ GPT 主张成立（OT 比较 = 似然差 + 类对常数）')
