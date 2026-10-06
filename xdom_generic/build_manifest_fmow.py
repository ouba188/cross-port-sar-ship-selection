"""FMoW 清单构建：遍历逐图 JSON → manifest（path, split, category, region, timestamp, country, lon, lat）。
区域 = 多边形质心经纬度的几何分区（自足，不引外部国家映射）。
"""
import csv
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

E = Path(r'E:/Hermes/xdomain_data/fmow/extracted')
OUT = Path(r'E:/Hermes/paper_tracking/xdom_generic/fmow_manifest.csv')

# ponytail: 大陆框（经度 -180..180, 纬度 -90..90）——FMoW 五大区对应的粗几何分区
BOXES = [
    ('Oceania',  110, 180, -50, 0),      # 澳新/太平洋
    ('Asia',      25, 180, 0, 80),
    ('Europe',   -25, 45, 35, 72),
    ('Africa',   -20, 52, -35, 37),
    ('Americas', -170, -30, -60, 75),
]


def region_of(lon, lat):
    for name, lo0, lo1, la0, la1 in BOXES:
        if lo0 <= lon <= lo1 and la0 <= lat <= la1:
            return name
    return 'Other'


def centroid(poly):
    pts = re.findall(r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', poly)
    if not pts:
        return None, None
    xs = [float(a) for a, _ in pts]; ys = [float(b) for _, b in pts]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def one(p):
    try:
        j = json.loads(Path(p).read_text(encoding='utf-8', errors='ignore'))
        bbs = j.get('bounding_boxes') or [{}]
        cat = (bbs[0].get('category') or '').strip().lower()
        loc = j.get('raw_location') or ''
        lon, lat = centroid(loc)
        ts = (j.get('timestamp') or '')[:10]
        img = Path(p).with_suffix('').with_suffix('.jpg')
        return [str(img), Path(p).parts[-5] if len(Path(p).parts) >= 5 else '', cat,
                region_of(lon, lat) if lon is not None else 'Other', ts,
                j.get('country_code', ''), f'{lon:.3f}' if lon is not None else '',
                f'{lat:.3f}' if lat is not None else '']
    except Exception:
        return None


def main():
    files = [str(p) for p in E.rglob('*_rgb.json')]
    print('json 总数', len(files), flush=True)
    rows, bad = [], 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(one, files, chunksize=200)):
            if r is None:
                bad += 1
            else:
                rows.append(r)
            if i % 50000 == 0:
                print('处理', i, flush=True)
    with OUT.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'split', 'category', 'region', 'date', 'country', 'lon', 'lat'])
        w.writerows(rows)
    print('写出', len(rows), '失败', bad, '->', OUT, flush=True)
    from collections import Counter
    print('region:', dict(Counter(r[3] for r in rows)), flush=True)
    print('category 数:', len(set(r[2] for r in rows)), flush=True)


if __name__ == '__main__':
    main()
