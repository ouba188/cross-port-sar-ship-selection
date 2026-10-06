"""SAR 23 港 头寸因果分解（HEADROOM_CAUSAL_PREREG_v2 的自家数据版）。
每港：meanH、win_rate、oracle/rank/rand 增益、capture。用主线真链（冻结 K=8 模块 + ratio2 键）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits

from bdh_pixel_loss_controls.conditional_likelihood import fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import CACHE, project_class_evidence_rt, src_weights, K, TAU0

OUT = Path(r'E:/Hermes/paper_tracking/sensitivity')
ANNOT = 64
BUDGET = 0.05          # 主线冻结预算 5%
SEED = 0


def run():
    rows = []
    for p in sorted(x.name[:-4] for x in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        base_logits = d['base_logits']; expert_logits = d['expert_logits']
        feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
        qpos = d['qpos']; spos = d['spos']; ranking = d['ranking']
        base = base_logits.argmax(1); expert = expert_logits.argmax(1)
        query_n = len(qpos)
        source_n = min(query_n, len(spos))
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
        # 逐行
        yq = y_ids[qpos]; bq = base[qpos]; eq = expert[qpos]
        bc = (bq == yq); ec = (eq == yq)
        D = (bq != eq)
        g = ec.astype(int) - bc.astype(int)
        win_rate = float((ec & ~bc)[D].mean()) if D.sum() else 0.0
        meanH = float((ec.astype(float) - bc.astype(float)).mean())
        k = min(int(BUDGET * query_n), int(D.sum()))
        base_acc = float(bc.mean())
        Dpos = np.flatnonzero(D)
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
        rows.append(dict(port=p, meanH=meanH, win_rate=win_rate, n_dis=int(D.sum()),
                         n_q=query_n, oracle_gain=oracle_gain, rank_gain=rank_gain,
                         rand_gain=rand_gain, capture=capture))
        print(f'{p}: meanH={meanH:+.4f} win={win_rate:.3f} dis={D.sum()} '
              f'oracle={oracle_gain:+.4f} rank={rank_gain:+.4f} rand={rand_gain:+.4f} cap={capture:.3f}', flush=True)
    (OUT / 'sar_causal_v2.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    from scipy.stats import spearmanr
    H = [r['meanH'] for r in rows]; W = [r['win_rate'] for r in rows]
    G = [r['rank_gain'] - r['rand_gain'] for r in rows]; C = [r['capture'] for r in rows]
    print('G vs meanH spearman:', spearmanr(G, H))
    print('G vs win_rate spearman:', spearmanr(G, W))
    print('capture 均值±std:', float(np.mean(C)), float(np.std(C)))
    print('capture:', [f'{c:.2f}' for c in C])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
