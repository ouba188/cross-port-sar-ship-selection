"""扩展缓存：给每折补存 Xs/Z（GBM 臂用真实特征）。幂等，只写 cache2/。"""
import csv
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
WS = Path(r'E:/临时会话/visual_reliable_baseline')
CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
CACHE2 = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache2')
CACHE2.mkdir(parents=True, exist_ok=True)


def main():
    from sklearn.decomposition import PCA
    rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8-sig')))
    R = np.load(ROOT / 'pooled_features.npy', mmap_mode='r').astype(np.float32)
    S = np.load(WS / 'b2_ssl_feats_38091.npz')['F'].astype(np.float32)
    ports_all = np.array([r['port'] for r in rows])
    mmsi = np.array([r['mmsi'] or '' for r in rows])
    product = np.array([r.get('product') or r.get('product_id') or '' for r in rows])
    for port in sorted(set(ports_all.tolist())):
        out = CACHE2 / f'{port}.npz'
        if out.exists():
            continue
        src = np.load(CACHE / f'{port}.npz', allow_pickle=False)
        if 'skipped' in src.files:
            np.savez_compressed(out, skipped=True, n=int(src['n']))
            continue
        te_mask = ports_all == port
        tr_mask = ~te_mask
        q_mmsi = set(mmsi[te_mask]) - {''}
        q_prod = set(product[te_mask]) - {''}
        tr_mask = tr_mask & ~np.isin(mmsi, list(q_mmsi)) & ~np.isin(product, list(q_prod))
        pca = PCA(n_components=128, random_state=0).fit(R[tr_mask])
        X = pca.transform(R)
        mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0) + 1e-6
        Xs = (X - mu) / sd
        pca2 = PCA(n_components=64, random_state=0).fit(S[tr_mask])
        Z = pca2.transform(S)
        np.savez_compressed(out, Xs=Xs.astype(np.float32), Z=Z.astype(np.float32))
        print('extended', port, flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
