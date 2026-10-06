"""真链跨域 PACS（FULLCHAIN_PACS_PREREG_v1）：K=7 泛化的主线 M1→M4 链，键=ratio2。
对照替身 pair-key 的 PACS 结果（P2 0/2）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.special import logsumexp
from sklearn.covariance import OAS
from threadpoolctl import threadpool_limits

from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import project_class_evidence_rt, src_weights

OUT = Path(r'E:/Hermes/paper_tracking/xdom_pacs')
TAU = 8.0
EM_STEPS = 128
PSEUDOCOUNT = 0.5
BUDGET = 0.2
SEEDS = list(range(5))
ANNOT = 64


def decision_features(base_logits, expert_logits, K):
    contrast = helmert(K, full=False)
    return np.concatenate([v @ contrast.T for v in (base_logits, expert_logits)], axis=1)


def fit_source(x, labels, ports, K):
    x, labels = np.asarray(x, dtype=np.float64), np.asarray(labels)
    assert x.shape == (len(labels), 2 * (K - 1)) and np.isfinite(x).all()
    assert np.array_equal(np.unique(labels), np.arange(K))
    w = src_weights(labels, ports)
    center = np.average(x, weights=w, axis=0)
    scale = np.maximum(np.sqrt(np.average((x - center) ** 2, weights=w, axis=0)), 1e-6)
    z = (x - center) / scale
    means = np.stack([np.average(z[labels == c], weights=w[labels == c], axis=0) for c in range(K)])
    residual = (z - means[labels]) * np.sqrt(len(x) * w[:, None])
    cov = OAS(assume_centered=True).fit(residual).covariance_
    np.linalg.cholesky(cov)
    return dict(center=center, scale=scale, means=means, covariance=cov,
                precision=np.linalg.solve(cov, np.eye(cov.shape[0])), weights=w)


def class_mass(loglikelihood, counts, K):
    loglikelihood, counts = np.asarray(loglikelihood), np.asarray(counts)
    assert loglikelihood.shape[1] == K and counts.shape == (K,)
    pseudo = counts + PSEUDOCOUNT
    prior = pseudo / pseudo.sum()
    for _ in range(EM_STEPS):
        joint = loglikelihood + np.log(prior)
        posterior = np.exp(joint - logsumexp(joint, axis=1)[:, None])
        prior = (posterior.sum(0) + pseudo) / (len(posterior) + pseudo.sum())
    return dict(prior=prior, posterior=posterior)


def robust_scores(ratio, base, expert, observed, K):
    supports = ((np.arange(1, 2 ** K)[:, None] >> np.arange(K)) & 1).astype(bool)
    supports = supports[np.all(supports[:, observed], axis=1)]
    r = ratio[np.arange(len(base)), expert]
    h = ratio[np.arange(len(base)), base]
    values = (supports[:, expert] * r - supports[:, base] * h) / supports.sum(1)[:, None]
    return values.min(0)


def macro_ba(pred, y, K):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(K)]))


def run():
    df = pd.read_parquet(r'E:/Hermes/xdomain_data/pacs/train-00000-of-00001.parquet')
    df['label'] = pd.factorize(df['label'])[0]
    K = int(df['label'].nunique())
    y = df['label'].values
    fA = np.load(OUT / 'featsA.npy')
    results = {}
    for td in sorted(df['domain'].unique()):
        fB = np.load(OUT / f'featsB_{td}.npy')
        te = (df['domain'] == td).values
        tr = ~te
        from sklearn.decomposition import PCA
        from sklearn.linear_model import RidgeClassifier
        pca = PCA(n_components=128, random_state=0).fit(fA[tr])
        X = pca.transform(fA)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs = (X - mu) / sd
        base_logits = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Xs[tr], y[tr]).decision_function(Xs)
        pcaB = PCA(n_components=64, random_state=0).fit(fB[tr])
        B = pcaB.transform(fB)
        mu2, sd2 = np.c_[Xs[tr], B[tr]].mean(0), np.c_[Xs[tr], B[tr]].std(0) + 1e-6
        XB = (np.c_[Xs, B] - mu2) / sd2
        expert_logits = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(XB[tr], y[tr]).decision_function(XB)
        feats = decision_features(base_logits, expert_logits, K)
        b = base_logits.argmax(1)
        e = expert_logits.argmax(1)
        yq, bq, eq = y[te], b[te], e[te]
        nq = len(yq)
        qpos = np.flatnonzero(te)
        spos = np.flatnonzero(tr)
        # 源行分层子样本（港×类均衡 → 这里按类均衡）
        rng_s = np.random.default_rng(1000)
        strat = {c: rng_s.choice(spos[y[spos] == c], min((y[spos] == c).sum(), max(1, nq // K)), replace=False)
                 for c in range(K)}
        source_pos = np.concatenate(list(strat.values()))
        ports_src = np.array(['src'] * len(source_pos))
        source_model = fit_source(feats[source_pos], y[source_pos], ports_src, K)
        rec = {'base': [], 'expert': [], 'rank': [], 'rand': [], 'thr': []}
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 1000)
            ann = rng.choice(qpos, ANNOT, replace=False)
            ay = y[ann]
            observed = np.bincount(ay, minlength=K) > 0
            p1 = shared_cov_predict(source_model, feats[ann], ay, feats[qpos])
            f1 = class_mass(p1['loglikelihood'], p1['counts'], K)
            ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
            s2 = robust_scores(ratio2, bq, eq, observed, K)
            # 源侧 s2 分位（非循环：对 source 行也跑一遍似然）
            p1s = shared_cov_predict(source_model, feats[ann], ay, feats[source_pos])
            ratio_src = project_class_evidence_rt(p1s['loglikelihood'], f1['prior'])
            s2src = robust_scores(ratio_src, b[source_pos], e[source_pos], observed, K)
            thr_q = np.quantile(s2src, 0.8)
            elig = np.flatnonzero(bq != eq)
            k = min(int(BUDGET * nq), len(elig))
            order = elig[np.argsort(-s2[elig])][:k]
            mask = np.zeros(nq, bool); mask[order] = True
            rec['base'].append(macro_ba(bq, yq, K))
            rec['expert'].append(macro_ba(eq, yq, K))
            rec['rank'].append(macro_ba(np.where(mask, eq, bq), yq, K))
            rsel = elig[rng.permutation(len(elig))[:k]]
            maskr = np.zeros(nq, bool); maskr[rsel] = True
            rec['rand'].append(macro_ba(np.where(maskr, eq, bq), yq, K))
            thrmask = (s2 >= thr_q) & (bq != eq)
            rec['thr'].append(macro_ba(np.where(thrmask, eq, bq), yq, K))
        results[td] = {k: float(np.mean(v)) for k, v in rec.items()}
        results[td]['n'] = int(nq)
        results[td]['dominates'] = bool(results[td]['expert'] >= results[td]['base'])
        print(td, json.dumps(results[td], ensure_ascii=False), flush=True)
    (OUT / 'fullchain_pacs_result.json').write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    doms = [d for d in results if results[d]['dominates']]
    P1 = float(np.mean([results[d]['dominates'] for d in results]))
    P2 = float(np.mean([results[d]['rank'] >= results[d]['rand'] for d in doms])) if doms else 0.0
    P3 = float(np.mean([results[d]['rank'] > results[d]['thr'] for d in doms])) if doms else 0.0
    print(json.dumps({'P1': P1, 'P2': P2, 'P3': P3, 'n_dom': len(doms)}, ensure_ascii=False))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
