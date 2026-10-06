"""FMoW CLIP 视图 B 特征提取（FMOW_CLIP_PREREG_v1）。对同一 27,557 子样本，CLIP ViT-B/32 图像编码。"""
import sys
from pathlib import Path
sys.path.insert(0, r'E:/Hermes/paper_tracking/xdom_generic')
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from threadpoolctl import threadpool_limits

import run_fmow as rf
OUT = Path(r'E:/Hermes/paper_tracking/xdom_generic')
BATCH = 128


def main():
    df = pd.read_csv(OUT / 'fmow_manifest.csv')
    s = rf.subsample(df)                 # 与 featsA_fmow_sub 对齐的子样本
    paths = list(s['image'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, pre = clip.load('ViT-B/32', device=dev)
    model.eval()
    feats = np.zeros((len(paths), 512), np.float32)
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            imgs = [pre(Image.open(p).convert('RGB')) for p in paths[i:i + BATCH]]
            x = torch.stack(imgs).to(dev)
            feats[i:i + BATCH] = model.encode_image(x).cpu().numpy()
            if i % (BATCH * 10) == 0:
                print('CLIP', i, '/', len(paths), flush=True)
    np.save(OUT / 'featsB_fmow_clip.npy', feats)
    print('DONE', feats.shape, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
