"""capture 驱动挖掘（CAPTURE_DRIVER_PREREG_v1）：每港测 s2 的 win/loss 判别力(AUROC) + capture + 候选驱动。"""
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
        yq = y_ids[qpos]; bq = base[qpos]; eq = expert[qpos]
        bc = (bq == yq); ec = (eq == yq)
        D = (bq != eq)
        g = ec.astype(int) - bc.astype(int)         # +1 win / 0 / -1 loss
        Dpos = np.flatnonzero(D)
        s2d = s2[D]; gd = g[D]
        win_loss = gd != 0
        # AUROC of s2 for win vs loss
        auroc_s2 = float(roc_auc_score((gd[win_loss] > 0).astype(int), s2d[win_loss])) if win_loss.sum() >= 2 and len(np.unique(gd[win_loss] > 0)) == 2 else float('nan')
        # expert 置信度校准（margin vs 对错，分歧行上）
        em = margin(expert_logits[qpos])[D]
        auroc_conf = float(roc_auc_score(ec[D].astype(int), em)) if len(np.unique(ec[D])) == 2 else float('nan')
        # capture（同 SAR_CAUSAL_V2）
        k = min(int(BUDGET * query_n), int(D.sum()))
        base_acc = float(bc.mean())
        order_o = Dpos[np.argsort(-g[Dpos])][:k]
        masko = np.zeros(query_n, bool); masko[order_o] = True
        oracle_gain = float((np.where(masko, eq, bq) == yq).mean()) - base_acc
        rng = np.random.default_rng(1)
        rsel = Dpos[rng.permutation(len(Dpos))[:k]]
        maskr = np.zeros(query_n, bool); maskr[rsel] = True
        rand_gain = float((np.where(maskr, eq, bq) == yq).mean()) - base_acc
        order_k = Dpos[np.argsort(-s2[Dpos])][:k]
        maskk = np.zeros(query_n, bool); maskk[order_k] = True
        rank_gain = float((np.where(maskk, eq, bq) == yq).mean()) - base_acc
        headroom = oracle_gain - rand_gain
        capture = (rank_gain - rand_gain) / headroom if headroom > 1e-9 else 0.0
        win_rate = float((ec & ~bc)[D].mean()) if D.sum() else 0.0
        rows.append(dict(port=p, capture=capture, auroc_s2=auroc_s2, auroc_conf=auroc_conf,
                         win_rate=win_rate, n_dis=int(D.sum()), base_acc=base_acc,
                         s2_std=float(s2d.std()), meanH=float((ec.astype(float) - bc.astype(float)).mean())))
        print(f'{p}: capture={capture:+.3f} AUROC_s2={auroc_s2:.3f} AUROC_conf={auroc_conf:.3f} '
              f'win={win_rate:.3f} dis={D.sum()} base={base_acc:.3f} s2std={s2d.std():.4f}', flush=True)
    (OUT / 'capture_driver.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    from scipy.stats import spearmanr
    R = rows
    def corr(a, b):
        aa = np.array([r[a] for r in R if not (isinstance(r[a], float) and (r[a] != r[a])) and not (isinstance(r[b], float) and (r[b] != r[b]))], dtype=float)
        bb = np.array([r[b] for r in R if not (isinstance(r[a], float) and (r[a] != r[a])) and not (isinstance(r[b], float) and (r[b] != r[b]))], dtype=float)
        return spearmanr(aa, bb)
    print('capture vs AUROC_s2:', corr('capture', 'auroc_s2'))
    for k in ['auroc_conf', 'win_rate', 'n_dis', 'base_acc', 's2_std', 'meanH']:
        print(f'AUROC_s2 vs {k}:', corr('auroc_s2', k))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
