"""PACS 下载（HF mirror，逐文件续传）。源 = figshare 备份的 pacs.zip 或 hf 镜像。
先试 hf-mirror 直连的 DomainBed 镜像（pacs 常用 hf 仓库）。
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = 'flwrlabs/pacs'
MIRROR = 'https://huggingface.co'
API = f'{MIRROR}/api/datasets/{REPO}?blobs=true'
BASE = f'{MIRROR}/datasets/{REPO}/resolve/main/'
D = Path('E:/Hermes/xdomain_data/pacs')
D.mkdir(parents=True, exist_ok=True)


def curl(*args):
    return subprocess.run(['curl', '--noproxy', '*', '-sL', '--retry', '4',
                           '--retry-delay', '4', '--max-time', '120', *args])


tmp = Path(tempfile.gettempdir()) / 'pacs_manifest.json'
meta = None
for attempt in range(5):
    curl('-o', str(tmp), API)
    try:
        meta = json.loads(tmp.read_text(encoding='utf-8'))
        if meta.get('siblings'):
            break
    except Exception:
        pass
    meta = None
if meta is None:
    print('✗ 清单获取失败', flush=True)
    sys.exit(3)
files = sorted((s['rfilename'], s.get('size') or 0)
               for s in meta['siblings'] if s['rfilename'].endswith(('.zip', '.tar.gz', '.tgz')))
if not files:
    files = sorted((s['rfilename'], s.get('size') or 0) for s in meta['siblings'])
print('文件 %d 个' % len(files), flush=True)
for name, size in files:
    dst = D / Path(name).name
    if dst.exists() and dst.stat().st_size == size:
        print('skip', name, flush=True)
        continue
    curl('-C', '-', '--max-time', '3600', '-o', str(dst), BASE + name)
    got = dst.stat().st_size if dst.exists() else 0
    print('%-40s %d / %d %s' % (name, got, size, 'OK' if got == size else 'INCOMPLETE'), flush=True)
print('DONE')
