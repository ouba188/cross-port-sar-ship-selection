"""margin 当键 vs s2 键（MARGIN_KEY_PREREG_v1）：测 expert 置信度是否键的信息来源。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score

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
        def capture_of(key_scores):
            order_k = Dpos[np.argsort(-key_scores[Dpos])][:k]
            maskk = np.zeros(query_n, bool); maskk[order_k] = True
            rg = float((np.where(maskk, eq, bq) == yq).mean()) - base_acc
            return (rg - rand_gain) / headroom if headroom > 1e-9 else 0.0
        cap_s2 = capture_of(s2)
        cap_mg = capture_of(mg)
        win_loss = g[D] != 0
        auroc_s2 = float(roc_auc_score((g[D][win_loss] > 0).astype(int), s2[D][win_loss])) if win_loss.sum() >= 2 else float('nan')
        auroc_mg = float(roc_auc_score((g[D][win_loss] > 0).astype(int), mg[D][win_loss])) if win_loss.sum() >= 2 else float('nan')
        auroc_conf = float(roc_auc_score(ec[D].astype(int), mg[D])) if len(np.unique(ec[D])) == 2 else float('nan')
        rows.append(dict(port=p, cap_s2=cap_s2, cap_mg=cap_mg, auroc_s2=auroc_s2,
                         auroc_mg=auroc_mg, auroc_conf=auroc_conf, n_dis=int(D.sum())))
        print(f'{p}: cap_s2={cap_s2:+.3f} cap_margin={cap_mg:+.3f} AUROC_s2={auroc_s2:.3f} '
              f'AUROC_margin={auroc_mg:.3f} AUROC_conf={auroc_conf:.3f}', flush=True)
    (OUT / 'margin_key.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    from scipy.stats import spearmanr
    def sc(a, b):
        aa = [r[a] for r in rows if r[a] == r[a]]; bb = [r[b] for r in rows if r[a] == r[a] and r[b] == r[b]]
        aa = [r[a] for r in rows if r[a] == r[a] and r[b] == r[b]]
        return spearmanr(aa, bb)
    print('capture_s2 vs capture_margin:', sc('cap_s2', 'cap_mg'))
    print('capture_margin vs AUROC_margin:', sc('cap_mg', 'auroc_mg'))
    print('mean cap_s2, mean cap_margin:', float(np.mean([r['cap_s2'] for r in rows])), float(np.mean([r['cap_mg'] for r in rows])))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
