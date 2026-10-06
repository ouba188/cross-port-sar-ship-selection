"""FMoW 跨区域协议（FMOW_PREREG_v2）：子样本 → 视图 A=R50(磁盘图) + 视图 B=采集元数据 → LOPO 5 区。
复用 run_xdom_generic.protocol（base/expert ridge、类别对键、20% 预算、三臂）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T
from PIL import Image

from run_xdom_generic import OUT, protocol

MAN = Path(r'E:/Hermes/paper_tracking/xdom_generic/fmow_manifest.csv')
BATCH = 128
CAP = 100
MIN_CELL = 25


def subsample(df):
    keep = []
    for (r, c), g in df.groupby(['region', 'category']):
        if r == 'Other' or len(g) < MIN_CELL:
            continue
        keep.append(g.sample(min(len(g), CAP), random_state=0))
    s = pd.concat(keep).reset_index(drop=True)
    return s


def meta_feats(df):
    """视图 B：采集元数据（禁用经纬度/国家）。"""
    d = pd.to_datetime(df['date'], errors='coerce')
    doy = d.dt.dayofyear.fillna(1).values
    year = d.dt.year.fillna(2000).values
    f = np.column_stack([(year - 2000) / 20.0, np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)])
    extra = []
    for i, p in enumerate(df['image']):
        jp = Path(str(p)).with_suffix('.json')
        try:
            j = json.loads(jp.read_text(encoding='utf-8', errors='ignore'))
            extra.append([float(j.get('cloud_cover', 0) or 0), float(j.get('gsd', 0) or 0),
                          float(j.get('sun_elevation_dbl', 0) or 0)])
        except Exception:
            extra.append([0.0, 0.0, 0.0])
    return np.c_[f, np.asarray(extra, dtype=np.float64)]


def feats_a(paths, tag, dev):
    f = OUT / f'featsA_fmow_{tag}.npy'
    if f.exists():
        return np.load(f)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V1).to(dev).eval()
    m.fc = nn.Identity()
    out = np.zeros((len(paths), 2048), np.float32)
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            xs = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + BATCH]]).to(dev)
            out[i:i + BATCH] = m(xs).cpu().numpy()
            if i % (BATCH * 10) == 0:
                print('A', i, '/', len(paths), flush=True)
    np.save(f, out)
    return out


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv(MAN)
    df = subsample(df)
    cats = sorted(df['category'].unique())
    df['label'] = df['category'].map({c: i for i, c in enumerate(cats)})
    df = df.reset_index(drop=True)
    print('子样本', len(df), '类', len(cats), '区', df['region'].nunique(), flush=True)
    print(df.groupby('region').size().to_dict(), flush=True)
    fA = feats_a(list(df['image']), 'sub', dev)
    fB = meta_feats(df)
    results = {}
    for td in sorted(df['region'].unique()):
        if (df['region'] == td).sum() < 50:
            print('跳过（测试样本 <50）', td, flush=True)
            continue
        results[td] = protocol(fA, fB, df, 'region', td, budget=0.05)
        print(td, json.dumps(results[td], ensure_ascii=False), flush=True)
    (OUT / 'fmow_result.json').write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    doms = list(results)
    dom = [d for d in doms if results[d]['dominates']]
    P1 = float(np.mean([results[d]['dominates'] for d in doms])) if doms else 0.0
    P2 = float(np.mean([results[d]['rank_pair'][0] >= results[d]['rand'][0] for d in dom])) if dom else 0.0
    P3 = float(np.mean([results[d]['rank_pair'][0] > results[d]['thr'][0] for d in dom])) if dom else 0.0
    print(json.dumps({'P1': P1, 'P2': P2, 'P3': P3, 'n_dom': len(dom)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
