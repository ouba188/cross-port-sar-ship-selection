"""H→G 因果律的跨数据集预测验证（留一数据集）。
从已有剂量-响应数据（PACS/FMoW 各 4 个 α 点）：
1) 定性律（G=0⇔H≈0、单调）是否两数据集都成立；
2) 定量律（斜率 c：G=c·H）是否可跨数据集外推——留一数据集拟合斜率、预测另一数据集 G。
判据：定性律两数据集都成立（可外推的"发现"）；定量斜率若外推误差大 ⇒ 律只在方向/阈值层可外推、幅度是数据集特定的。
"""
import json
import numpy as np

D = json.load(open(r'E:/Hermes/paper_tracking/xdom_generic/headroom_doseresponse.json'))
pacs = [(r['H'], r['G']) for r in D['pacs']]
fmow = [(r['H'], r['G']) for r in D['fmow']]


def fit_slope(pts):
    H = np.array([p[0] for p in pts]); G = np.array([p[1] for p in pts])
    c = float((H @ G) / (H @ H)) if (H @ H) > 0 else 0.0
    return c


def predict(c, H):
    return c * np.array(H)


def report(src_name, src, tgt_name, tgt):
    c = fit_slope(src)
    Ht = np.array([p[0] for p in tgt]); Gt = np.array([p[1] for p in tgt])
    Gp = predict(c, Ht)
    mae = float(np.mean(np.abs(Gp - Gt)))
    # 方向正确率：G 是否 >0 与预测是否 >0 一致
    sign_ok = float(np.mean((Gp > 0) == (Gt > 0)))
    return dict(fit_slope=c, mae=mae, sign_acc=sign_ok, Gt=list(Gt), Gp=list(Gp))


r1 = report('pacs', pacs, 'fmow', fmow)
r2 = report('fmow', fmow, 'pacs', pacs)

# 定性律：每个数据集 G=0 当且仅当 H≈0
def qualitative(pts, name):
    signs = [(p[0] > 0.01, p[1] > 0.001) for p in pts]
    ok = all((h == g) for h, g in signs)
    return name, ok, signs

q1 = qualitative(pacs, 'pacs')
q2 = qualitative(fmow, 'fmow')

out = dict(
    pacs_slope=r1['fit_slope'], fmow_slope=r2['fit_slope'],
    slope_ratio=r1['fit_slope'] / r2['fit_slope'] if r2['fit_slope'] else None,
    predict_fmow_from_pacs=r1, predict_pacs_from_fmow=r2,
    qualitative_pacs=q1, qualitative_fmow=q2)
print(json.dumps(out, ensure_ascii=False, indent=1))
