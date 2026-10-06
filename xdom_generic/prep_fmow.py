"""FMoW 数据准备（FMOW_OFFICEHOME_PREREG_v1）：解包 → manifest → R50 特征 + 时间元数据特征。
用法：python prep_fmow.py --inspect（只看结构）| --build（解包+提特征）
"""
import argparse
import csv
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

D = Path(r'E:/Hermes/xdomain_data/fmow')
OUT = Path(r'E:/Hermes/paper_tracking/xdom_generic')
OUT.mkdir(parents=True, exist_ok=True)


def inspect():
    for name in ('train-images.tar.gz', 'train-metadata.tar.gz', 'val-images.tar.gz', 'val-metadata.tar.gz'):
        p = D / name
        if not p.exists():
            print(f'{name}: 缺失', flush=True)
            continue
        print(f'== {name} ({p.stat().st_size/1e9:.2f} GB)', flush=True)
        try:
            with tarfile.open(p, 'r:gz') as tf:
                members = []
                for i, m in enumerate(tf):
                    members.append(m.name)
                    if i >= 8:
                        break
                print('  样例成员:', members, flush=True)
        except Exception as e:
            print('  读取失败:', repr(e), flush=True)


def build():
    """解包两组 tar 并输出 manifest（image_path,label,region,timestamp）。"""
    ex = D / 'extracted'
    ex.mkdir(exist_ok=True)
    for name in ('train-images.tar.gz', 'train-metadata.tar.gz', 'val-images.tar.gz', 'val-metadata.tar.gz'):
        p = D / name
        if not p.exists():
            print('缺', name, flush=True)
            continue
        marker = ex / (name + '.done')
        if marker.exists():
            continue
        print('解包', name, flush=True)
        with tarfile.open(p, 'r:gz') as tf:
            tf.extractall(ex)
        marker.write_text('ok')
    # 找元数据表（csv/json）
    metas = [p for p in ex.rglob('*') if p.suffix.lower() in ('.csv', '.json', '.jsonl', '.txt')]
    print('元数据候选:', [str(p.relative_to(ex)) for p in metas[:10]], flush=True)
    print('图像目录候选:', [str(p.relative_to(ex)) for p in list(ex.rglob("*"))[:0]], flush=True)
    for p in metas[:3]:
        try:
            if p.suffix.lower() == '.csv':
                rows = list(csv.DictReader(p.open(encoding='utf-8', errors='ignore')))
                print(p.name, '列:', list(rows[0].keys()) if rows else '空', '行:', len(rows), flush=True)
                print('  样例:', rows[0] if rows else {}, flush=True)
            else:
                txt = p.read_text(encoding='utf-8', errors='ignore')[:400]
                print(p.name, '预览:', txt, flush=True)
        except Exception as e:
            print(p.name, '读取失败', repr(e), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--inspect', action='store_true')
    ap.add_argument('--build', action='store_true')
    a = ap.parse_args()
    if a.inspect or not a.build:
        inspect()
    if a.build:
        build()
