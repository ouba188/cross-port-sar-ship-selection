"""PACS 分歧键诊断（S0 级）：源端 key 拟合 AUC、分歧率、key 类别组成、逐域分歧结构。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

OUT = Path(r'E:/Hermes/paper_tracking/xdom_pacs')
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_parquet(r'E:/Hermes/xdomain_data/pacs/train-00000-of-00001.parquet',
                         engine='pyarrow')
    fA = np.load(OUT / 'featsA.npy')
    y = df.label.values
    res = {}
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
        key_src = (b[tr] != y[tr]) & (e[tr] == y[tr])
        # 源端 out-of-fold 拟合质量（防 in-sample 虚高）
        lr = LogisticRegression(max_iter=2000)
        pred = cross_val_predict(lr, Xs[tr], key_src.astype(int), cv=3, method='predict_proba')[:, 1]
        src_auc = roc_auc_score(key_src, pred) if key_src.sum() > 0 and (~key_src).sum() > 0 else float('nan')
        # 目标域分歧结构
        te_key = (b[te] != y[te]) & (e[te] == y[te])
        dis_rate = float((b[te] != e[te]).mean())
        res[td] = dict(src_key_rate=float(key_src.mean()), src_key_auc=src_auc,
                       tgt_disagreement_rate=dis_rate,
                       tgt_key_rate=float(te_key.mean()),
                       base_ba=float(np.mean([(b[te][y[te] == c] == c).mean() for c in np.unique(y)])),
                       expert_ba=float(np.mean([(e[te][y[te] == c] == c).mean() for c in np.unique(y)])))
        print(td, json.dumps(res[td], ensure_ascii=False))
    (OUT / 'key_diagnosis.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
