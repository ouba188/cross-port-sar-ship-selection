"""PACS 类别对键修复（PACS_PAIRKEY_PATCH_PREREG_v1）：rank_pair vs rank_lr vs rand vs thr。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import cross_val_predict

OUT = Path(r'E:/Hermes/paper_tracking/xdom_pacs')
K = 7


def main():
    df = pd.read_parquet(r'E:/Hermes/xdomain_data/pacs/train-00000-of-00001.parquet',
                         engine='pyarrow')
    fA = np.load(OUT / 'featsA.npy')
    y = df.label.values
    results = {}
    for td in sorted(df.domain.unique()):
        fB = np.load(OUT / f'featsB_{td}.npy')
        te = (df.domain == td).values
        tr = ~te
        pca = PCA(n_components=128, random_state=0).fit(fA[tr])
        X = pca.transform(fA)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs = (X - mu) / sd
        mb = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Xs[tr], y[tr])
        base = mb.decision_function(Xs)
        pca2 = PCA(n_components=64, random_state=0).fit(fB[tr])
        Z = pca2.transform(fB)
        Mtr = np.c_[Xs[tr], Z[tr]]
        mu2, sd2 = Mtr.mean(0), Mtr.std(0) + 1e-6
        me = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Mtr - mu2) / sd2, y[tr])
        expert = me.decision_function((np.c_[Xs, Z] - mu2) / sd2)
        b = base.argmax(1); e = expert.argmax(1)
        # 源侧类别对键（同主线：按类均衡加权）
        w = np.zeros(len(tr), dtype=np.float64)
        for c in np.unique(y[tr]):
            w[tr & (y == c)] = 1.0 / np.unique(y[tr]).size / (y[tr] == c).sum()
        cond = np.bincount(K * K * y[tr] + K * b[tr] + e[tr], weights=w[tr],
                           minlength=K ** 3).reshape(K, K, K)
        # P(专家对 | 基座错, 类别对)：pair 条件质量 = cond[c,p,c]/sum_e cond[c,p,:]，取该行 (c, b) 对
        pair_score = np.zeros(K * K)
        for c in range(K):
            for p in range(K):
                row = cond[c, p, :]
                denom = row.sum()
                # 键 = 基座预测 p ≠ c 时专家对的率
                pair_score[K * c + p] = (cond[c, p, c] / denom) if denom > 0 else 0.0
        spair = pair_score[K * y[te] + b[te]]
        # 旧 LR 键（对照）
        key_src = (b[tr] != y[tr]) & (e[tr] == y[tr])
        lr = LogisticRegression(max_iter=2000)
        lr_pred = cross_val_predict(lr, Xs[tr], key_src.astype(int), cv=3, method='predict_proba')[:, 1]
        lr = LogisticRegression(max_iter=2000).fit(Xs[tr], key_src.astype(int))
        slr = lr.predict_proba(Xs[te])[:, 1]
        bq, eq, yq = b[te], e[te], y[te]
        eligible = np.flatnonzero(bq != eq)
        k = min(int(0.2 * len(yq)), len(eligible))
        rng = np.random.default_rng(0)

        def macro(pred):
            return float(np.mean([(pred[yq == c] == c).mean() for c in np.unique(yq)]))
        res = {}
        for name, score in (('rank_pair', spair), ('rank_lr', slr)):
            order = eligible[np.argsort(-score[eligible])]
            mask = order[:k]
            pred = np.where(np.isin(np.arange(len(yq)), mask), eq, bq)
            res[name] = macro(pred)
        rand_mask = rng.choice(eligible, k, replace=False)
        pred = np.where(np.isin(np.arange(len(yq)), rand_mask), eq, bq)
        res['rand'] = macro(pred)
        thr = np.quantile(lr.predict_proba(Xs[tr])[:, 1], 0.8)
        thr_mask = eligible[slr[eligible] >= thr][:k]
        pred = np.where(np.isin(np.arange(len(yq)), thr_mask), eq, bq)
        res['thr'] = macro(pred)
        results[td] = res
        print(td, json.dumps(res, ensure_ascii=False))
    (OUT / 'pacs_pairkey_v1.json').write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    doms = ['cartoon', 'sketch']
    p2 = [results[d]['rank_pair'] >= results[d]['rand'] for d in doms]
    p3 = [results[d]['rank_pair'] > results[d]['thr'] for d in doms]
    print('P2pair 支配域:', sum(p2), '/2 | P3 支配域:', sum(p3), '/2')


if __name__ == '__main__':
    main()
