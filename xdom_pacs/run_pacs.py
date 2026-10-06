"""PACS 跨域泛化管线（PACS_PREREG_v1）：视图 A=ImageNet R50；视图 B=源域监督 R18。
LOPO 4 域；协议 = XDOMAIN_PREREG_v1 §2（基座 ridge(A)/专家 ridge([A|B])、20% 预算、秩 vs 随机 vs 源标定阈值、macro）。
断点续跑：markers/ 下每折标记。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T

OUT = Path(r'E:/Hermes/paper_tracking/xdom_pacs')
MARK = OUT / 'markers'
MARK.mkdir(parents=True, exist_ok=True)
BATCH = 128
EPOCHS = 12
LR = 1e-3


def img_bytes(b):
    return b['bytes'] if isinstance(b, dict) else b

def load():
    df = pd.read_parquet(r'E:/Hermes/xdomain_data/pacs/train-00000-of-00001.parquet',
                         engine='pyarrow')
    return df


@torch.no_grad()
def extract_a(df):
    """视图 A：ImageNet R50 2048。"""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tf = T.Compose([T.Resize(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V1).to(dev).eval()
    m.fc = nn.Identity()
    from PIL import Image
    from io import BytesIO
    feats = np.zeros((len(df), 2048), np.float32)
    for i in range(0, len(df), BATCH):
        chunk = df['image'].iloc[i:i + BATCH]
        xs = torch.stack([tf(Image.open(BytesIO(img_bytes(b))).convert('RGB')) for b in chunk]).to(dev)
        feats[i:i + BATCH] = m(xs).cpu().numpy()
        if i % (BATCH * 8) == 0:
            print('A', i, '/', len(df), flush=True)
    np.save(OUT / 'featsA.npy', feats)
    return feats


def train_b(df, target_domain):
    """视图 B：在 3 个源域上训练 R18（监督）。"""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tf = T.Compose([T.RandomHorizontalFlip(), T.Resize(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    src = df[df.domain != target_domain].reset_index(drop=True)
    m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(512, df.label.nunique())
    m = m.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    from PIL import Image
    from io import BytesIO
    imgs = [Image.open(BytesIO(img_bytes(b))).convert('RGB') for b in src['image']]
    y = torch.tensor(src['label'].values)
    n = len(imgs)
    for ep in range(EPOCHS):
        m.train()
        perm = torch.randperm(n)
        tot = 0; loss_sum = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xs = torch.stack([tf(imgs[j]) for j in idx]).to(dev)
            out = m(xs)
            loss = crit(out, y[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += len(idx); loss_sum += float(loss) * len(idx)
        print(f'B({target_domain}) ep {ep} loss {loss_sum / tot:.4f}', flush=True)
    m.eval()
    m.fc = nn.Identity()
    feats = np.zeros((len(df), 512), np.float32)
    tf2 = T.Compose([T.Resize(224), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        for i in range(0, len(df), BATCH):
            xs = torch.stack([tf2(imgs_all[j]) for j in range(i, min(i + BATCH, len(df)))]).to(dev)
            feats[i:i + len(xs)] = m(xs).cpu().numpy()
    np.save(OUT / f'featsB_{target_domain}.npy', feats)
    return feats


def protocol(fA, fB, df, target_domain):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeClassifier
    te = (df.domain == target_domain).values
    tr = ~te
    y = df.label.values
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
    yq = y[te]; bq = b[te]; eq = e[te]
    # 分歧键：源侧拟合 1[base 错 ∧ expert 对]
    src_b = b[tr]; src_e = e[tr]; src_y = y[tr]
    key = (src_b != src_y) & (src_e == src_y)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000).fit(Xs[tr], key.astype(int))
    score = lr.predict_proba(Xs[te])[:, 1]
    # 预算 20%：按秩 vs 随机 vs 源标定阈值
    k = max(1, int(0.2 * len(yq)))
    eligible = np.flatnonzero(bq != eq)
    k = min(k, len(eligible))
    rng = np.random.default_rng(0)
    order = eligible[np.argsort(-score[eligible])]
    rank_mask = order[:k]
    rand_mask = rng.choice(eligible, k, replace=False)
    thr = np.quantile(lr.predict_proba(Xs[tr])[:, 1], 0.8)
    thr_mask = eligible[score[eligible] >= thr][:k]
    def macro(y, pred):
        return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))
    def wacc(y, pred):
        return float((pred == y).mean())
    res = {'base': (macro(yq, bq), wacc(yq, bq)),
           'expert_all': (macro(yq, eq), wacc(yq, eq))}
    for name, mask in (('rank', rank_mask), ('rand', rand_mask), ('thr', thr_mask)):
        pred = np.where(np.isin(np.arange(len(yq)), mask), eq, bq)
        res[name] = (macro(yq, pred), wacc(yq, pred))
    res['dominates'] = res['expert_all'][0] >= res['base'][0]
    res['n'] = int(te.sum())
    return res


def main():
    df = load()
    print('domains:', dict(df.domain.value_counts()), 'labels:', df.label.nunique(), flush=True)
    fA = np.load(OUT / 'featsA.npy') if (OUT / 'featsA.npy').exists() else extract_a(df)
    from PIL import Image
    from io import BytesIO
    global imgs_all
    imgs_all = [Image.open(BytesIO(img_bytes(b))).convert('RGB') for b in df['image']]
    results = {}
    for td in sorted(df.domain.unique()):
        rmark = MARK / f'{td}.json'
        if rmark.exists():
            results[td] = json.loads(rmark.read_text(encoding='utf-8'))
            print('skip', td, flush=True)
            continue
        fB = np.load(OUT / f'featsB_{td}.npy') if (OUT / f'featsB_{td}.npy').exists() else train_b(df, td)
        res = protocol(fA, fB, df, td)
        results[td] = res
        rmark.write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
        print(td, json.dumps(res, ensure_ascii=False), flush=True)
    (OUT / 'pacs_result_v1.json').write_text(json.dumps(results, ensure_ascii=False), encoding='utf-8')
    # 汇总：P1/P2/P3/P5
    doms = list(results.keys())
    p1 = float(np.mean([results[d]['dominates'] for d in doms]))
    rank_ba = [results[d]['rank'][0] for d in doms]
    rand_ba = [results[d]['rand'][0] for d in doms]
    thr_ba = [results[d]['thr'][0] for d in doms]
    p2 = float(np.mean([r > rn for r, rn in zip(rank_ba, rand_ba)]))
    p3 = float(np.mean([r > t for r, t in zip(rank_ba, thr_ba)]))
    summ = {'P1_dominance': p1, 'P2_rank_ge_rand': p2, 'P3_rank_gt_thr': p3,
            'rank_ba': rank_ba, 'rand_ba': rand_ba, 'thr_ba': thr_ba,
            'base_ba': [results[d]['base'][0] for d in doms],
            'expert_ba': [results[d]['expert_all'][0] for d in doms]}
    (OUT / 'pacs_summary_v1.json').write_text(json.dumps(summ, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
