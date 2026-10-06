"""CLIP 是否真互补的诊断：CLIP 单独精度 + 错误重叠（R50 错处 CLIP 对不对）。"""
import sys
from pathlib import Path
sys.path.insert(0, r'E:/Hermes/paper_tracking/xdom_generic')
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier

import run_fmow as rf
OUT = Path(r'E:/Hermes/paper_tracking/xdom_generic')


def main():
    df = pd.read_csv(OUT / 'fmow_manifest.csv')
    s = rf.subsample(df)
    top8 = s['category'].value_counts().index[:8]
    mask = s['category'].isin(top8).values
    s8 = s[mask].reset_index(drop=True)
    y = pd.factorize(s8['category'])[0].astype(int)
    dom = s8['region'].values
    fA = np.load(OUT / 'featsA_fmow_sub.npy')[mask]
    fB = np.load(OUT / 'featsB_fmow_clip.npy')[mask]
    K = 8
    for td in sorted(np.unique(dom)):
        te = (dom == td); tr = ~te
        pcaA = PCA(n_components=128, random_state=0).fit(fA[tr]); X = pcaA.transform(fA)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xs = (X - mu) / sd
        pcaB = PCA(n_components=64, random_state=0).fit(fB[tr]); B = pcaB.transform(fB)
        mu2, sd2 = np.c_[Xs[tr], B[tr]].mean(0), np.c_[Xs[tr], B[tr]].std(0) + 1e-6
        base = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Xs[tr], y[tr]).predict(Xs[te])
        clip = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(B[tr], y[tr]).predict(B[te])
        expert = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((np.c_[Xs[tr], B[tr]] - mu2) / sd2, y[tr]).predict((np.c_[Xs[te], B[te]] - mu2) / sd2)
        yt = y[te]
        acc = lambda p: float((p == yt).mean())
        # 错误重叠：R50 错的行里 CLIP 对的比例（互补的核心证据）
        a_wrong = base != yt
        clip_saves = float((clip == yt)[a_wrong].mean()) if a_wrong.sum() else float('nan')
        # 反之 CLIP 错 R50 对
        c_wrong = clip != yt
        r50_saves = float((base == yt)[c_wrong].mean()) if c_wrong.sum() else float('nan')
        print(f'{td}: base={acc(base):.4f} CLIP_alone={acc(clip):.4f} expert={acc(expert):.4f} | R50错处CLIP对率={clip_saves:.3f} CLIP错处R50对率={r50_saves:.3f}', flush=True)


if __name__ == '__main__':
    main()
