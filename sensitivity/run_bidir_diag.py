"""双向选模型（100% 预算）诊断：键按符号决定每个分歧样本用 base 还是 expert。
对比 fusion(恒 expert) 与 oracle(按真值挑对)。"""
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
ANNOT = 64


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def wacc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def margin(logits):
    s = np.sort(logits, axis=1)
    return s[:, -1] - s[:, -2]


rows = []
for p in sorted(x.name[:-4] for x in CACHE.glob('*.npz')):
    d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
    if 'skipped' in d.files:
        continue
    feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
    qpos = d['qpos']; spos = d['spos']
    base = d['base_logits'].argmax(1); expert = d['expert_logits'].argmax(1)
    expert_logits = d['expert_logits']
    query_n = len(qpos); source_n = min(query_n, len(spos))
    rng_s = np.random.default_rng(1000)
    strat = {c: rng_s.choice(spos[y_ids[spos] == c],
                             max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False)
             for c in np.unique(y_ids[spos])}
    source_pos = np.concatenate(list(strat.values()))
    yq = y_ids[qpos]; bq = base[qpos]; eq = expert[qpos]
    bc = (bq == yq); ec = (eq == yq)
    D = (bq != eq)
    # 键 s2（符号 = 双向）
    ann = np.random.default_rng(0).choice(qpos, ANNOT, replace=False)
    ay = y_ids[ann]
    observed = np.bincount(ay, minlength=K) > 0
    sm = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
    p1 = shared_cov_predict(sm, feats[ann], ay, feats[qpos])
    f1 = class_mass(p1['loglikelihood'], p1['counts'])
    ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
    s2 = robust_scores(ratio2, bq, eq, observed)
    mg = margin(expert_logits[qpos])
    mgz = (mg - mg.mean()) / (mg.std() + 1e-9)
    s2z = (s2 - s2.mean()) / (s2.std() + 1e-9)
    ens = s2z + mgz
    # 臂（100% 预算双向：每个分歧样本按符号选；非分歧=base/expert 同）
    fusion_pred = eq
    s2_bidir = np.where(D, np.where(s2 > 0, eq, bq), bq)
    ens_bidir = np.where(D, np.where(ens > 0, eq, bq), bq)
    # oracle 双向：分歧样本挑对的那个
    oracle = np.where(D, np.where(ec, eq, bq), bq)  # ec True→expert对；False→base对
    rows.append(dict(port=p,
                     ba_base=macro_ba(yq, bq), ba_fusion=macro_ba(yq, eq),
                     ba_s2=macro_ba(yq, s2_bidir), ba_ens=macro_ba(yq, ens_bidir),
                     ba_oracle=macro_ba(yq, oracle),
                     acc_fusion=wacc(yq, eq), acc_s2=wacc(yq, s2_bidir),
                     acc_ens=wacc(yq, ens_bidir), acc_oracle=wacc(yq, oracle),
                     n_dis=int(D.sum()), n_expert_wrong_base_right=int(((ec == False) & (bc == True)).sum())))
    print(f'{p}: fusion={macro_ba(yq,eq):.3f} s2={macro_ba(yq,s2_bidir):.3f} '
          f'ens={macro_ba(yq,ens_bidir):.3f} oracle={macro_ba(yq,oracle):.3f} '
          f'(expert错base对 {rows[-1]["n_expert_wrong_base_right"]} 个)', flush=True)

import json
Path('bidir_diag.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
for k in ['ba_base', 'ba_fusion', 'ba_s2', 'ba_ens', 'ba_oracle']:
    print(f'{k:11s} = {np.mean([r[k] for r in rows]):.4f}')
for k in ['acc_fusion', 'acc_s2', 'acc_ens', 'acc_oracle']:
    print(f'{k:11s} = {np.mean([r[k] for r in rows]):.4f}')
print('专家错基座对(可被双向挽回)总均值 =', np.mean([r['n_expert_wrong_base_right'] for r in rows]))
