"""FMoW 下载完成守望：轮询两个图像包达到预期体积 → 解包 + 结构报告 → 写完成标记。
不自动跑协议（元数据格式未知，等结构报告后再接管线）。
"""
import subprocess
import sys
import time
from pathlib import Path

D = Path(r'E:/Hermes/xdomain_data/fmow')
OUT = Path(r'E:/Hermes/paper_tracking/xdom_generic')
PY = r'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe'
EXPECT = {'train-images.tar.gz': 9.5e9, 'val-images.tar.gz': 1.6e9,
          'train-metadata.tar.gz': 2.5e8, 'val-metadata.tar.gz': 4.0e7}
MARK = OUT / 'fmow_extract_done.marker'
LOG = OUT / 'fmow_watch.log'


def log(msg):
    with LOG.open('a', encoding='utf-8') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')


def ready():
    for name, need in EXPECT.items():
        p = D / name
        if not p.exists() or p.stat().st_size < need * 0.95:
            return False, name, (p.stat().st_size if p.exists() else 0), need
    return True, None, None, None


def main():
    while True:
        ok, name, got, need = ready()
        if ok:
            log('四个文件就位，开始解包')
            r = subprocess.run([PY, str(OUT / 'prep_fmow.py'), '--build'],
                               capture_output=True, text=True)
            log('prep 输出:\n' + (r.stdout or '')[-4000:])
            log('prep 错误:\n' + (r.stderr or '')[-2000:])
            MARK.write_text('ok')
            print('FMOW_PREP_DONE', flush=True)
            return
        log(f'等待 {name}: {got/1e9:.2f}/{need/1e9:.2f} GB')
        time.sleep(300)


if __name__ == '__main__':
    main()
