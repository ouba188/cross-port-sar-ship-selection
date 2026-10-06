"""互补键集成（ENSEMBLE_KEY_PREREG_v1）：s2 ∪ margin z-score 求和，对比单键与 oracle 上界。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits

from bdh_pixel_loss_controls.conditional_likelihood import fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import CACHE, project_class_evidence_rt, K

OUT = Path(r'E:/Hermes/paper_tracking/sensitivity')
ANNOT = 64
BUDGET = 0.05
SEED = 0


def margin(logits):
    s = np.sort(logits, axis=1)
    return s[:, -1] - s[:, -2]


def run():
    rows = []
    for p in sorted(x.name[:-4] for x in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        base_logits = d['base_logits']; expert_logits = d['expert_logits']
        feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
        qpos = d['qpos']; spos = d['spos']
        base = base_logits.argmax(1); expert = expert_logits.argmax(1)
        query_n = len(qpos); source_n = min(query_n, len(spos))
        rng_s = np.random.default_rng(1000)
        strat = {c: rng_s.choice(spos[y_ids[spos] == c],
                                 max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False)
                 for c in np.unique(y_ids[spos])}
        source_pos = np.concatenate(list(strat.values()))
        source_model = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
        ann = np.random.default_rng(SEED * 1000).choice(qpos, ANNOT, replace=False)
        ay = y_ids[ann]
        observed = np.bincount(ay, minlength=K) > 0
        p1 = shared_cov_predict(source_model, feats[ann], ay, feats[qpos])
        f1 = class_mass(p1['loglikelihood'], p1['counts'])
        ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
        s2 = robust_scores(ratio2, base[qpos], expert[qpos], observed)
        mg = margin(expert_logits[qpos])
        yq = y_ids[qpos]; bq = base[qpos]; eq = expert[qpos]
        bc = (bq == yq); ec = (eq == yq)
        D = (bq != eq)
        g = ec.astype(int) - bc.astype(int)
        Dpos = np.flatnonzero(D)
        k = min(int(BUDGET * query_n), int(D.sum()))
        base_acc = float(bc.mean())
        order_o = Dpos[np.argsort(-g[Dpos])][:k]
        masko = np.zeros(query_n, bool); masko[order_o] = True
        oracle_gain = float((np.where(masko, eq, bq) == yq).mean()) - base_acc
        rng = np.random.default_rng(1)
        rsel = Dpos[rng.permutation(len(Dpos))[:k]]
        maskr = np.zeros(query_n, bool); maskr[rsel] = True
        rand_gain = float((np.where(maskr, eq, bq) == yq).mean()) - base_acc
        headroom = oracle_gain - rand_gain
        def capture_of(key):
            order_k = Dpos[np.argsort(-key[Dpos])][:k]
            maskk = np.zeros(query_n, bool); maskk[order_k] = True
            rg = float((np.where(maskk, eq, bq) == yq).mean()) - base_acc
            return (rg - rand_gain) / headroom if headroom > 1e-9 else 0.0
        cap_s2 = capture_of(s2)
        cap_mg = capture_of(mg)
        # 集成：分歧行上 z-score 求和
        zs2 = (s2 - s2[D].mean()) / (s2[D].std() + 1e-12)
        zmg = (mg - mg[D].mean()) / (mg[D].std() + 1e-12)
        cap_ens = capture_of(zs2 + zmg)
        cap_pick = max(cap_s2, cap_mg)
        rows.append(dict(port=p, cap_s2=cap_s2, cap_mg=cap_mg, cap_ens=cap_ens, cap_pick=cap_pick))
        print(f'{p}: s2={cap_s2:+.3f} margin={cap_mg:+.3f} ens={cap_ens:+.3f} pick={cap_pick:+.3f}', flush=True)
    (OUT / 'ensemble_key.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    ms2 = float(np.mean([r['cap_s2'] for r in rows])); mmg = float(np.mean([r['cap_mg'] for r in rows]))
    mens = float(np.mean([r['cap_ens'] for r in rows])); mpick = float(np.mean([r['cap_pick'] for r in rows]))
    print(f'mean s2={ms2:.3f} margin={mmg:.3f} ensemble={mens:.3f} pick={mpick:.3f}')
    print(f'ensemble/pick = {mens/mpick:.3f}' if mpick > 0 else 'pick<=0')
    print('ensemble 超单键均值:', mens > max(ms2, mmg))
    # 逐港：ensemble 是否在 s2 差的港补上
    improved = sum(1 for r in rows if r['cap_ens'] > max(r['cap_s2'], r['cap_mg']) - 1e-9)
    print(f'ensemble 严格超两单键的港数: {improved}/{len(rows)}')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
