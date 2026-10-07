"""端口外预测 + 相对收益可靠性模型：4 对照复跑（回答"4.54pp 空间能否用可部署信息识别"）。
训练数据：源港 leave-one-port-out 预测（从 pred_cache 组装：pred_cache[p'] 的 qpos 样本=排除 p' 训练的模型输出）。
学习目标：Δ = 1(e=y) - 1(b=y) ∈ {-1,0,+1}（两错=0）。按港×类平衡权重。
对照：simple gate / s2 / 端口外 LR（二分类）/ 端口外相对收益（ridge on Δ）。
收缩：源模型与 64标签模型线性混合（λ 固定 0.5）。
用法：python run_outofport_reliability.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, r'E:/Hermes/paper_tracking/sensitivity')

import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LogisticRegression, Ridge

from bdh_pixel_loss_controls.conditional_likelihood import fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as shared_cov_predict
from run_adaptive import K, project_class_evidence_rt

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
ANNOT = 64
SEEDS = [0, 1, 2, 3, 4]
LAMBDA = 0.5


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def load_q(port):
    d = np.load(CACHE / f'{port}.npz', allow_pickle=False)
    q = d['qpos']
    return d['feats'][q], d['y_ids'][q], d['base_logits'][q].argmax(1), d['expert_logits'][q].argmax(1)


def balance_weights(y, port):
    w = np.zeros(len(y), dtype=np.float64)
    for pp in np.unique(port):
        pm = port == pp
        for c in np.unique(y[pm]):
            cm = pm & (y == c)
            w[cm] = 1.0 / cm.sum()
    return w / w.sum() * len(w)


def run(smoke=False):
    all_ports = sorted(x.name[:-4] for x in CACHE.glob('*.npz'))
    all_ports = [p for p in all_ports if p != 'Melbourne']
    DATA = {p: load_q(p) for p in all_ports}

    rows = []
    for p in all_ports:
        src_ports = [q for q in all_ports if q != p]
        X_tr, Y_tr, B_tr, E_tr, P_tr = [], [], [], [], []
        for q in src_ports:
            fx, fy, fb, fe = DATA[q]
            X_tr.append(fx); Y_tr.append(fy); B_tr.append(fb); E_tr.append(fe)
            P_tr.append(np.full(len(fy), q))
        X_tr = np.concatenate(X_tr); Y_tr = np.concatenate(Y_tr)
        B_tr = np.concatenate(B_tr); E_tr = np.concatenate(E_tr); P_tr = np.concatenate(P_tr)
        DELTA_tr = (E_tr == Y_tr).astype(int) - (B_tr == Y_tr).astype(int)
        w_tr = balance_weights(Y_tr, P_tr)
        disp = DELTA_tr != 0
        X_d, y_d, w_d = X_tr[disp], (DELTA_tr[disp] > 0).astype(int), w_tr[disp]

        lr = LogisticRegression(max_iter=2000).fit(X_d, y_d, sample_weight=w_d)
        rr = Ridge(alpha=1.0).fit(X_tr, DELTA_tr, sample_weight=w_tr)

        fx_t, fy_t, fb_t, fe_t = DATA[p]
        qn = len(fy_t); D = (fb_t != fe_t)
        row = {'port': p, 'sg': [], 's2': [], 'lr': [], 'rr': []}
        for seed in SEEDS:
            ann = np.random.default_rng(seed).choice(qn, ANNOT, replace=False)
            ay = fy_t[ann]
            ev = np.ones(qn, bool); ev[ann] = False
            observed = np.bincount(ay, minlength=K) > 0

            d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
            feats_all = d['feats']; y_all = d['y_ids']; ports_all = d['ports_all']
            qpos = d['qpos']; spos = d['spos']
            ann_full = qpos[ann]  # qpos 相对 → 全量索引
            bq = d['base_logits'][qpos].argmax(1); eq = d['expert_logits'][qpos].argmax(1); yq = y_all[qpos]
            source_n = min(qn, len(spos))
            rng_s = np.random.default_rng(1000)
            strat = {c: rng_s.choice(spos[y_all[spos] == c], max(1, int(round(source_n * (y_all[spos] == c).sum() / len(spos)))), replace=False) for c in np.unique(y_all[spos])}
            source_pos = np.concatenate(list(strat.values()))
            sm = fit_source(feats_all[source_pos], y_all[source_pos], ports_all[source_pos])
            r_ind = shared_cov_predict(sm, feats_all[ann_full], ay, feats_all[qpos])
            llq = r_ind['loglikelihood']

            switch_sg = D & observed[eq] & (llq[np.arange(qn), eq] > llq[np.arange(qn), bq])
            g_sg = np.where(switch_sg, eq, bq)
            row['sg'].append(macro_ba(yq[ev], g_sg[ev]))

            f1 = class_mass(llq, r_ind['counts'])
            ratio2 = project_class_evidence_rt(llq, f1['prior'])
            s2 = robust_scores(ratio2, bq, eq, observed)
            g_s2 = np.where(D, np.where(s2 > 0, eq, bq), bq)
            row['s2'].append(macro_ba(yq[ev], g_s2[ev]))

            d_ann = (fe_t[ann] == ay).astype(int) - (fb_t[ann] == ay).astype(int)
            disp_ann = d_ann != 0
            if disp_ann.sum() >= 2 and 0 < (d_ann[disp_ann] > 0).mean() < 1:
                lr64 = LogisticRegression(max_iter=2000).fit(fx_t[ann][disp_ann], (d_ann[disp_ann] > 0).astype(int))
                coef_lr = ((1 - LAMBDA) * lr.coef_ + LAMBDA * lr64.coef_).ravel()
                int_lr = float((1 - LAMBDA) * lr.intercept_[0] + LAMBDA * lr64.intercept_[0])
                lr_score = fx_t @ coef_lr + int_lr
            else:
                lr_score = lr.decision_function(fx_t)
            g_lr = np.where(D, np.where(lr_score > 0, fe_t, fb_t), fb_t)
            row['lr'].append(macro_ba(fy_t[ev], g_lr[ev]))

            rr64 = Ridge(alpha=1.0).fit(fx_t[ann], d_ann)
            coef_rr = (1 - LAMBDA) * rr.coef_ + LAMBDA * rr64.coef_
            int_rr = (1 - LAMBDA) * rr.intercept_ + LAMBDA * rr64.intercept_
            rr_score = fx_t @ coef_rr + int_rr
            g_rr = np.where(D, np.where(rr_score > 0, fe_t, fb_t), fb_t)
            row['rr'].append(macro_ba(fy_t[ev], g_rr[ev]))

        rows.append(row)
        print(f'{p}: sg={np.mean(row["sg"]):.4f} s2={np.mean(row["s2"]):.4f} '
              f'lr={np.mean(row["lr"]):.4f} rr={np.mean(row["rr"]):.4f}', flush=True)
        if smoke:
            break

    json.dump(rows, open('outofport_reliability.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    from scipy.stats import wilcoxon
    print('\n=== BA 汇总 ===')
    for k in ['sg', 's2', 'lr', 'rr']:
        v = np.array([np.mean(x[k]) for x in rows])
        print(f'{k}: {v.mean():.4f}')
    def wil(a, b):
        va = np.array([np.mean(x[a]) for x in rows]); vb = np.array([np.mean(x[b]) for x in rows])
        dd = va - vb; w = wilcoxon(dd)
        print(f'{a} vs {b}: Δ={dd.mean():+.4f} 正港{int((dd>0).sum())}/{len(dd)} p={w.pvalue:.4f}')
    print('\n=== 关键对照（相对 simple gate）===')
    wil('lr', 'sg'); wil('rr', 'sg'); wil('s2', 'sg')
    wil('lr', 's2'); wil('rr', 's2')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
