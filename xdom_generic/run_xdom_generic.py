"""跨域通用管线（FMOW_OFFICEHOME_PREREG_v1）：officehome / fmow 两数据集。
协议同 PACS：视图 A=R50-ImageNet；视图 B=源域监督 R18（officehome）/ 时间元数据（fmow）；
base=ridge(A)、expert=ridge([A|B])；类别对键；20% 预算；rank_pair / rand / thr 三臂；macro BA。
"""
import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier

OUT = Path(r'E:/Hermes/paper_tracking/xdom_generic')
OUT.mkdir(parents=True, exist_ok=True)
BATCH = 128


def img_bytes(b):
    return b['bytes'] if isinstance(b, dict) else b


def load_officehome():
    D = Path(r'E:/Hermes/xdomain_data/office_home')
    frames = [pd.read_parquet(p) for p in sorted(D.glob('*.parquet'))]
    df = pd.concat(frames, ignore_index=True)
    df['label'] = df['label'].astype(int)
    df = df.reset_index(drop=True)
    return df, 'domain'


def r50_feats(images_bytes, tag, dev):
    f = OUT / f'featsA_{tag}.npy'
    if f.exists():
        return np.load(f)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V1).to(dev).eval()
    m.fc = nn.Identity()
    out = np.zeros((len(images_bytes), 2048), np.float32)
    with torch.no_grad():
        for i in range(0, len(images_bytes), BATCH):
            xs = torch.stack([tf(Image.open(BytesIO(img_bytes(b))).convert('RGB'))
                              for b in images_bytes[i:i + BATCH]]).to(dev)
            out[i:i + BATCH] = m(xs).cpu().numpy()
            if i % (BATCH * 8) == 0:
                print('A', i, '/', len(images_bytes), flush=True)
    np.save(f, out)
    return out


def train_b(df, target_domain, dom_col, dev, tag):
    f = OUT / f'featsB_{tag}_{target_domain}.npy'
    if f.exists():
        return np.load(f)
    src = df[df[dom_col] != target_domain].reset_index(drop=True)
    tf_aug = T.Compose([T.RandomHorizontalFlip(), T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tf_eval = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    n_cls = df['label'].nunique()
    m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(512, n_cls)
    m = m.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    src_bytes = list(src['image'].values)          # ponytail: 只留字节，按批解码（全解码会 OOM）
    y = torch.tensor(src['label'].values)
    n = len(src_bytes)
    for ep in range(12):
        m.train(); perm = torch.randperm(n); tot = 0; lsum = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xs = torch.stack([tf_aug(Image.open(BytesIO(img_bytes(src_bytes[int(j)]))).convert('RGB'))
                              for j in idx]).to(dev)
            loss = crit(m(xs), y[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += len(idx); lsum += float(loss) * len(idx)
        print(f'B({target_domain}) ep{ep} loss {lsum/tot:.4f}', flush=True)
    m.eval(); m.fc = nn.Identity()
    all_bytes = list(df['image'].values)
    out = np.zeros((len(df), 512), np.float32)
    with torch.no_grad():
        for i in range(0, len(df), BATCH):
            xs = torch.stack([tf_eval(Image.open(BytesIO(img_bytes(all_bytes[j]))).convert('RGB'))
                              for j in range(i, min(i + BATCH, len(df)))]).to(dev)
            out[i:i + len(xs)] = m(xs).cpu().numpy()
    np.save(f, out)
    return out


def protocol(fA, fB, df, dom_col, target, budget=0.2):
    K = int(df['label'].nunique())
    y = df['label'].values
    te = (df[dom_col] == target).values
    tr = ~te
    pca = PCA(n_components=128, random_state=0).fit(fA[tr])
    X = pca.transform(fA)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    mb = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Xs[tr], y[tr])
    base = mb.decision_function(Xs)
    Z = fB
    mu2, sd2 = np.c_[Xs[tr], Z[tr]].mean(0), np.c_[Xs[tr], Z[tr]].std(0) + 1e-6
    me = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((np.c_[Xs[tr], Z[tr]] - mu2) / sd2, y[tr])
    expert = me.decision_function((np.c_[Xs, Z] - mu2) / sd2)
    b = base.argmax(1); e = expert.argmax(1)
    # 源侧类别对键（同主线）
    w = np.zeros(len(tr), dtype=np.float64)
    for c in np.unique(y[tr]):
        w[tr & (y == c)] = 1.0 / np.unique(y[tr]).size / (y[tr] == c).sum()
    cond = np.bincount(K * K * y[tr] + K * b[tr] + e[tr], weights=w[tr],
                       minlength=K ** 3).reshape(K, K, K)
    pair = np.zeros(K * K)
    for c in range(K):
        for p in range(K):
            den = cond[c, p, :].sum()
            pair[K * c + p] = cond[c, p, c] / den if den > 0 else 0.0
    yq, bq, eq = y[te], b[te], e[te]
    spair = pair[K * yq + bq]
    spair_src = pair[K * y[tr] + b[tr]]   # 阈值取源侧分位（目标侧不可用）
    eligible = np.flatnonzero(bq != eq)
    k = min(int(budget * len(yq)), len(eligible))

    def macro(pred):
        return float(np.mean([(pred[yq == c] == c).mean() for c in np.unique(yq)]))

    def wacc(pred):
        return float((pred == yq).mean())

    rng = np.random.default_rng(0)
    res = {'base': (macro(bq), wacc(bq)), 'expert': (macro(eq), wacc(eq))}
    order = eligible[np.argsort(-spair[eligible])]
    for name, mask in (('rank_pair', order[:k]),
                       ('rand', rng.choice(eligible, k, replace=False) if k else np.array([], int)),
                       ('thr', eligible[spair[eligible] >= np.quantile(spair_src, 0.8)][:k])):
        pred = np.where(np.isin(np.arange(len(yq)), mask), eq, bq)
        res[name] = (macro(pred), wacc(pred))
    res['dominates'] = bool(res['expert'][0] >= res['base'][0])
    res['n'] = int(te.sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['officehome', 'fmow'])
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.dataset == 'officehome':
        df, dom_col = load_officehome()
        fA = r50_feats(list(df['image']), 'officehome', dev)
        results = {}
        for td in sorted(df[dom_col].unique()):
            fB = train_b(df, td, dom_col, dev, 'officehome')
            results[td] = protocol(fA, fB, df, dom_col, td)
            print(td, json.dumps(results[td], ensure_ascii=False), flush=True)
    else:
        print('fmow 需先解包建 manifest（见 prep_fmow.py）', flush=True)
        sys.exit(2)
    (OUT / f'{a.dataset}_result.json').write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    doms = list(results)
    p3 = float(np.mean([results[d]['rank_pair'][0] > results[d]['thr'][0] for d in doms]))
    p2 = float(np.mean([results[d]['rank_pair'][0] >= results[d]['rand'][0] for d in doms]))
    print(json.dumps({'P1_dominance': float(np.mean([results[d]['dominates'] for d in doms])),
                      'P2_rank_ge_rand': p2, 'P3_rank_gt_thr': p3}, ensure_ascii=False))


if __name__ == '__main__':
    main()
