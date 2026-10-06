"""头寸→选择增益 剂量-响应（HEADROOM_CAUSAL_PREREG_v1）。K 泛化链（PACS K=7 / FMoW K=8）。
B_α = α·B + (1−α)·N；测 H(α)=expert−base、G(α)=rank−rand。
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

from run_pacs_fullchain import decision_features, fit_source, class_mass, robust_scores, macro_ba
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import project_class_evidence_rt
from run_xdom_fullchain import load_fmow_clip

OUT = Path(r'E:/Hermes/paper_tracking/xdom_generic')
ALPHAS = [0.0, 1 / 3, 2 / 3, 1.0]
ANNOT = 64
BUDGET = 0.2
SEEDS = list(range(5))


def run_chain_k(fA, fB_map, y, dom, name, K):
    y = y.astype(int)
    res = {}
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
        rec = {k: [] for k in ('base', 'expert', 'rank', 'rand')}
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 1000)
            ann = rng.choice(qpos, ANNOT, replace=False); ay = y[ann]
            observed = np.bincount(ay, minlength=K) > 0
            p1 = shared_cov_predict(source_model, feats[ann], ay, feats[qpos])
            f1 = class_mass(p1['loglikelihood'], p1['counts'], K)
            ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
            s2 = robust_scores(ratio2, bq, eq, observed, K)
            elig = np.flatnonzero(bq != eq)
            k = min(int(BUDGET * nq), len(elig))
            order = elig[np.argsort(-s2[elig])][:k]
            mask = np.zeros(nq, bool); mask[order] = True
            rec['base'].append(macro_ba(bq, yq, K)); rec['expert'].append(macro_ba(eq, yq, K))
            rec['rank'].append(macro_ba(np.where(mask, eq, bq), yq, K))
            rsel = elig[rng.permutation(len(elig))[:k]]
            maskr = np.zeros(nq, bool); maskr[rsel] = True
            rec['rand'].append(macro_ba(np.where(maskr, eq, bq), yq, K))
        res[td] = {k: float(np.mean(v)) for k, v in rec.items()}
        res[td]['n'] = int(nq)
    return res


def blend_map(fB_map, alpha, seed=0):
    rng = np.random.default_rng(seed)
    return {td: alpha * B + (1 - alpha) * rng.normal(0, B.std(0) + 1e-6, size=B.shape) for td, B in fB_map.items()}


def one_dataset(fA, fB_map, y, dom, name, targets, K):
    rows = []
    for alpha in ALPHAS:
        res = run_chain_k(fA, blend_map(fB_map, alpha), y, dom, f'{name}_a{alpha:.2f}', K)
        Hs = [res[td]['expert'] - res[td]['base'] for td in targets if td in res]
        Gs = [res[td]['rank'] - res[td]['rand'] for td in targets if td in res]
        rows.append(dict(alpha=alpha, H=float(np.mean(Hs)), G=float(np.mean(Gs)), Hd=Hs, Gd=Gs))
        print(f'{name} a={alpha:.2f} H={rows[-1]["H"]:+.4f} G={rows[-1]["G"]:+.4f} Gd={[f"{g:+.4f}" for g in Gs]}', flush=True)
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
    targets_pacs = [d for d in fB_map if 'sketch' in d or 'cartoon' in d]
    rows_pacs = one_dataset(fA, fB_map, y, dom, 'pacs', targets_pacs, K_pacs)

    fA2, fB_map2, y2, dom2, _ = load_fmow_clip()
    rows_fmow = one_dataset(fA2, fB_map2, y2, dom2, 'fmowclip', ['Asia', 'Europe'], 8)

    (OUT / 'headroom_doseresponse.json').write_text(json.dumps(dict(pacs=rows_pacs, fmow=rows_fmow),
                                                             ensure_ascii=False, indent=1), encoding='utf-8')
    from scipy.stats import spearmanr
    allH = [r['H'] for ds in (rows_pacs, rows_fmow) for r in ds]
    allG = [r['G'] for ds in (rows_pacs, rows_fmow) for r in ds]
    rho, p = spearmanr(allH, allG)
    print(json.dumps({'spearman_H_G': float(rho), 'p': float(p)}, ensure_ascii=False))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
