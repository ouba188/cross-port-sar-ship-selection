"""真链跨域复验（XDOM_FULLCHAIN_PREREG_v1）：FMoW / Office-Home，K=8 前 8 高频类，主线真链（ratio2 键）。
用冻结 K=8 模块原样（decision_features/fit_source/class_mass/robust_scores/shared_cov）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')
sys.path.insert(0, r'E:/Hermes/paper_tracking/xdom_generic')

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from bdh_pixel_loss_controls.conditional_likelihood import (
    decision_features, fit_source, class_mass, robust_scores)
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import project_class_evidence_rt, src_weights

OUT = Path(r'E:/Hermes/paper_tracking/xdom_generic')
K = 8
BUDGET = 0.2
ANNOT = 64
SEEDS = list(range(5))


def run_chain(fA, fB_map, y, dom, name):
    """K=8 主线真链；fB_map: {target_domain -> (n,d) 视图B}（OH 逐折，FMoW 常量）。"""
    y = y.astype(int)
    doms = sorted(np.unique(dom))
    res = {}
    for td in doms:
        te = (dom == td)
        tr = ~te
        fB = np.asarray(fB_map[td], dtype=np.float64)
        assert fB.shape[0] == len(y) and fB.ndim == 2
        pca = PCA(n_components=128, random_state=0).fit(fA[tr])
        X = pca.transform(fA)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs = (X - mu) / sd
        base_logits = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Xs[tr], y[tr]).decision_function(Xs)
        if fB.shape[1] > 64:
            B = PCA(n_components=64, random_state=0).fit(fB[tr]).transform(fB)
        else:
            B = fB
        mu2, sd2 = np.c_[Xs[tr], B[tr]].mean(0), np.c_[Xs[tr], B[tr]].std(0) + 1e-6
        XB = (np.c_[Xs, B] - mu2) / sd2
        expert_logits = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(XB[tr], y[tr]).decision_function(XB)
        feats = decision_features(base_logits, expert_logits)
        b = base_logits.argmax(1); e = expert_logits.argmax(1)
        yq, bq, eq = y[te], b[te], e[te]
        nq = len(yq); qpos = np.flatnonzero(te); spos = np.flatnonzero(tr)
        rng_s = np.random.default_rng(1000)
        strat = {c: rng_s.choice(spos[y[spos] == c], min((y[spos] == c).sum(), max(1, nq // K)), replace=False)
                 for c in range(K)}
        source_pos = np.concatenate(list(strat.values()))
        ports_src = np.array(['src'] * len(source_pos))
        source_model = fit_source(feats[source_pos], y[source_pos], ports_src)
        rec = {k: [] for k in ('base', 'expert', 'rank', 'rand', 'thr')}
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 1000)
            ann = rng.choice(qpos, ANNOT, replace=False)
            ay = y[ann]
            observed = np.bincount(ay, minlength=K) > 0
            p1 = shared_cov_predict(source_model, feats[ann], ay, feats[qpos])
            f1 = class_mass(p1['loglikelihood'], p1['counts'])
            ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
            s2 = robust_scores(ratio2, bq, eq, observed)
            p1s = shared_cov_predict(source_model, feats[ann], ay, feats[source_pos])
            s2src = robust_scores(project_class_evidence_rt(p1s['loglikelihood'], f1['prior']),
                                  b[source_pos], e[source_pos], observed)
            thr_q = np.quantile(s2src, 0.8)
            elig = np.flatnonzero(bq != eq)
            k = min(int(BUDGET * nq), len(elig))
            order = elig[np.argsort(-s2[elig])][:k]
            mask = np.zeros(nq, bool); mask[order] = True
            def mb(pred):
                return float(np.mean([(pred[yq == c] == c).mean() for c in range(K)]))
            rec['base'].append(mb(bq)); rec['expert'].append(mb(eq))
            rec['rank'].append(mb(np.where(mask, eq, bq)))
            rsel = elig[rng.permutation(len(elig))[:k]]
            maskr = np.zeros(nq, bool); maskr[rsel] = True
            rec['rand'].append(mb(np.where(maskr, eq, bq)))
            thrmask = (s2 >= thr_q) & (bq != eq)
            rec['thr'].append(mb(np.where(thrmask, eq, bq)))
        r = {k: float(np.mean(v)) for k, v in rec.items()}
        r['n'] = int(nq); r['dominates'] = bool(r['expert'] >= r['base'])
        res[td] = r
        print(f'{name}/{td}', json.dumps(r, ensure_ascii=False), flush=True)
    doms_ok = [d for d in res if res[d]['dominates']]
    P1 = float(np.mean([res[d]['dominates'] for d in res]))
    P2 = float(np.mean([res[d]['rank'] >= res[d]['rand'] for d in doms_ok])) if doms_ok else 0.0
    P3 = float(np.mean([res[d]['rank'] > res[d]['thr'] for d in doms_ok])) if doms_ok else 0.0
    print(json.dumps({'dataset': name, 'P1': P1, 'P2': P2, 'P3': P3, 'n_dom': len(doms_ok)}, ensure_ascii=False))
    return res


def load_fmow():
    import run_fmow as rf
    df = pd.read_csv(r'E:/Hermes/paper_tracking/xdom_generic/fmow_manifest.csv')
    s = rf.subsample(df)
    fA = np.load(OUT / 'featsA_fmow_sub.npy')
    top8 = s['category'].value_counts().index[:8]
    mask = s['category'].isin(top8).values
    s8 = s[mask].reset_index(drop=True)
    fA8 = fA[mask]
    fB8 = rf.meta_feats(s8)
    y = pd.factorize(s8['category'])[0].astype(int)
    dom = s8['region'].values
    fB_map = {td: fB8 for td in sorted(np.unique(dom))}
    return fA8, fB_map, y, dom, 'fmow'


def load_fmow_clip():
    import run_fmow as rf
    df = pd.read_csv(r'E:/Hermes/paper_tracking/xdom_generic/fmow_manifest.csv')
    s = rf.subsample(df)
    fA = np.load(OUT / 'featsA_fmow_sub.npy')
    fB = np.load(OUT / 'featsB_fmow_clip.npy')
    top8 = s['category'].value_counts().index[:8]
    mask = s['category'].isin(top8).values
    fA8 = fA[mask]
    fB8 = fB[mask]
    y = pd.factorize(s.loc[mask, 'category'])[0].astype(int)
    dom = s.loc[mask, 'region'].values
    fB_map = {td: fB8 for td in sorted(np.unique(dom))}
    return fA8, fB_map, y, dom, 'fmowclip'


def load_oh():
    import run_xdom_generic as rg
    df, _ = rg.load_officehome()
    df['label'] = df['label'].astype(int)
    fA = np.load(OUT / 'featsA_officehome.npy')
    top8 = df['label'].value_counts().index[:8]
    mask = df['label'].isin(top8).values
    df8 = df[mask].reset_index(drop=True)
    fA8 = fA[mask]
    dom = df8['domain'].values
    fB_map = {td: np.load(OUT / f'featsB_officehome_{td}.npy')[mask] for td in sorted(np.unique(dom))}
    y = pd.factorize(df8['label'])[0].astype(int)
    return fA8, fB_map, y, dom, 'officehome'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True, choices=['fmow', 'officehome', 'fmowclip'])
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        if a.dataset == 'fmow':
            fA, fB_map, y, dom, name = load_fmow()
        elif a.dataset == 'fmowclip':
            fA, fB_map, y, dom, name = load_fmow_clip()
        else:
            fA, fB_map, y, dom, name = load_oh()
        res = run_chain(fA, fB_map, y, dom, name)
        (OUT / f'fullchain_{name}_result.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
