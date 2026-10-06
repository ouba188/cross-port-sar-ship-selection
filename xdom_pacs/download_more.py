"""FMoW(WILDS) + Office-Home 下载（curl 逐片续传）。"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

TARGETS = [
    ('jbourcier/fmow-rgb-baseline', 'E:/Hermes/xdomain_data/fmow', ()),
    ('flwrlabs/office-home', 'E:/Hermes/xdomain_data/office_home', ()),
]


def fetch(repo, dest, prefixes):
    API = f'https://huggingface.co/api/datasets/{repo}?blobs=true'
    BASE = f'https://huggingface.co/datasets/{repo}/resolve/main/'
    D = Path(dest); D.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / f'{repo.replace("/", "_")}.json'
    meta = None
    for _ in range(5):
        subprocess.run(['curl', '--noproxy', '*', '-sL', '--retry', '4', '--retry-delay', '4',
                        '--max-time', '90', '-o', str(tmp), API])
        try:
            meta = json.loads(tmp.read_text(encoding='utf-8'))
            if meta.get('siblings'):
                break
        except Exception:
            pass
        meta = None
    if meta is None:
        print(f'✗ {repo} 清单失败', flush=True); return
    files = [(s['rfilename'], s.get('size') or 0) for s in meta['siblings']
             if s['rfilename'].endswith(('.parquet', '.zip', '.tar.gz'))]
    if prefixes:
        files = [f for f in files if any(p in f[0] for p in prefixes)]
    total = sum(z for _, z in files)
    print(f'{repo}: {len(files)} 片 / {total/1e9:.2f} GB', flush=True)
    for name, size in files:
        dst = D / Path(name).name
        if dst.exists() and dst.stat().st_size == size:
            continue
        if dst.exists() and dst.stat().st_size < 1000:
            dst.unlink()   # LFS 指针/错误页残留，删掉重下
        for attempt in range(4):
            if dst.exists():
                subprocess.run(['curl', '--noproxy', '*', '-sL', '-C', '-', '--retry', '5',
                                '--retry-delay', '5', '--max-time', '2400', '-o', str(dst),
                                BASE + name + '?download=true'])
            else:
                subprocess.run(['curl', '--noproxy', '*', '-sL', '--retry', '5',
                                '--retry-delay', '5', '--max-time', '2400', '-o', str(dst),
                                BASE + name + '?download=true'])
            if dst.exists() and dst.stat().st_size > 1000 and (not size or dst.stat().st_size == size):
                break
            if dst.exists() and dst.stat().st_size < 1000:
                dst.unlink()
        got = dst.stat().st_size if dst.exists() else 0
        print('%-36s %10d / %10d %s' % (name[:36], got, size, 'OK' if got == size else 'INCOMPLETE'), flush=True)
    print(f'{repo} DONE', flush=True)


for repo, dest, prefixes in TARGETS:
    fetch(repo, dest, prefixes)
print('ALL_DONE')
