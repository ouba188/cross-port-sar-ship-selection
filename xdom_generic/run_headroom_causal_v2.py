"""头寸因果重做（HEADROOM_CAUSAL_PREREG_v2）：逐行分解 + 定位效率。
对 PACS(4域)/FMoW(5区,CLIP) 每域测：mean H、win_rate(可恢复信号)、oracle/rank/rand 增益、capture(定位效率)。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/Hermes/paper_tracking/xdom_pacs')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')
sys.path.insert(0, r'E:/Hermes/paper_tracking/xdom_generic')
sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from run_pacs_fullchain import decision_features, fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import project_class_evidence_rt
from run_xdom_fullchain import load_fmow_clip

OUT = Path(r'E:/Hermes/paper_tracking/xdom_generic')
ANNOT = 64
BUDGET = 0.2
SEEDS = list(range(5))


def diagnose(fA, fB_map, y, dom, K, name):
    y = y.astype(int)
    rows = []
    for td in sorted(np.unique(dom)):
        te = (dom == td); tr = ~te
        fB = np.asarray(fB_map[td], dtype=np.float64)
        pca = PCA(n_components=128, random_state=0).fit(fA[tr]); X = pca.transform(fA)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xs = (X - mu) / sd
        base_logits = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Xs[tr], y[tr]).decision_function(Xs)
        B = PCA(n_components=64, random_state=0).fit(fB[tr]).transform(fB) if fB.shape[1] > 64 else fB
        mu2, sd2 = np.c_[Xs[tr], B[tr]].mean(0), np.c_[Xs[tr], B[tr]].std(0) + 1e-6
        XB = (np.c_[Xs, B] - mu2) / sd2
        expert_logits = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(XB[tr], y[tr]).decision_function(XB)
        feats = decision_features(base_logits, expert_logits, K)
        b = base_logits.argmax(1); e = expert_logits.argmax(1)
        yq, bq, eq = y[te], b[te], e[te]
        nq = len(yq); qpos = np.flatnonzero(te); spos = np.flatnonzero(tr)
        rng_s = np.random.default_rng(1000)
        strat = {c: rng_s.choice(spos[y[spos] == c], min((y[spos] == c).sum(), max(1, nq // K)), replace=False)
                 for c in range(K)}
        source_pos = np.concatenate(list(strat.values()))
        source_model = fit_source(feats[source_pos], y[source_pos], np.array(['src'] * len(source_pos)), K)
        # 逐行量（用 seed 0 的 64 标签算一次键；oracle/win 用真标签，非部署）
        ann = np.random.default_rng(0).choice(qpos, ANNOT, replace=False); ay = y[ann]
        observed = np.bincount(ay, minlength=K) > 0
        p1 = shared_cov_predict(source_model, feats[ann], ay, feats[qpos])
        f1 = class_mass(p1['loglikelihood'], p1['counts'], K)
        ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
        s2 = robust_scores(ratio2, bq, eq, observed, K)
        # 逐行
        bc = (bq == yq); ec = (eq == yq)
        D = (bq != eq)
        g = ec.astype(int) - bc.astype(int)   # +1 换对 / 0 / -1 换错
        win_rate = float((ec & ~bc)[D].mean()) if D.sum() else 0.0
        meanH = float((ec.astype(float) - bc.astype(float)).mean())
        k = min(int(BUDGET * nq), int(D.sum()))
        def acc(pred): return float((pred == yq).mean())
        base_acc = acc(bq)
        # oracle：按 g 降序选 top-k 分歧行切换
        Dpos = np.flatnonzero(D)
        order_o = Dpos[np.argsort(-g[Dpos])][:k]
        masko = np.zeros(nq, bool); masko[order_o] = True
        oracle_gain = acc(np.where(masko, eq, bq)) - base_acc
        # rand
        rng = np.random.default_rng(1)
        rsel = Dpos[rng.permutation(len(Dpos))[:k]]
        maskr = np.zeros(nq, bool); maskr[rsel] = True
        rand_gain = acc(np.where(maskr, eq, bq)) - base_acc
        # key
        order_k = Dpos[np.argsort(-s2[Dpos])][:k]
        maskk = np.zeros(nq, bool); maskk[order_k] = True
        rank_gain = acc(np.where(maskk, eq, bq)) - base_acc
        headroom = oracle_gain - rand_gain
        capture = (rank_gain - rand_gain) / headroom if headroom > 1e-9 else 0.0
        rows.append(dict(domain=td, meanH=meanH, win_rate=win_rate, n_dis=int(D.sum()),
                         oracle_gain=oracle_gain, rank_gain=rank_gain, rand_gain=rand_gain,
                         capture=capture))
        print(f'{name}/{td}: meanH={meanH:+.4f} win_rate={win_rate:.3f} oracle={oracle_gain:+.4f} '
              f'rank={rank_gain:+.4f} rand={rand_gain:+.4f} capture={capture:.3f}', flush=True)
    return rows


def main():
    pacs = pd.read_parquet(r'E:/Hermes/xdomain_data/pacs/train-00000-of-00001.parquet')
    pacs['label'] = pd.factorize(pacs['label'])[0]
    mask = pacs['label'].isin(pacs['label'].value_counts().index[:8]).values
    pacs8 = pacs[mask].reset_index(drop=True)
    y = pd.factorize(pacs8['label'])[0].astype(int)
    dom = pacs8['domain'].values
    K_pacs = int(y.max()) + 1
    fA = np.load(Path(r'E:/Hermes/paper_tracking/xdom_pacs/featsA.npy'))[mask]
    fB_map = {td: np.load(Path(r'E:/Hermes/paper_tracking/xdom_pacs') / f'featsB_{td}.npy')[mask]
              for td in sorted(np.unique(dom))}
    r_pacs = diagnose(fA, fB_map, y, dom, K_pacs, 'pacs')

    fA2, fB_map2, y2, dom2, _ = load_fmow_clip()
    r_fmow = diagnose(fA2, fB_map2, y2, dom2, 8, 'fmow')

    (OUT / 'headroom_causal_v2.json').write_text(json.dumps(dict(pacs=r_pacs, fmow=r_fmow),
                                                          ensure_ascii=False, indent=1), encoding='utf-8')
    from scipy.stats import spearmanr, pearsonr
    allr = r_pacs + r_fmow
    H = [r['meanH'] for r in allr]; W = [r['win_rate'] for r in allr]; G = [r['rank_gain'] - r['rand_gain'] for r in allr]
    C = [r['capture'] for r in allr]
    print('G vs meanH spearman:', spearmanr(G, H))
    print('G vs win_rate spearman:', spearmanr(G, W))
    print('capture 均值±std:', float(np.mean(C)), float(np.std(C)))
    print('capture 列表:', [f'{c:.2f}' for c in C])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
