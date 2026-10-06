"""双向键提升（KEY_IMPROVE_PREREG_v1）。6 键候选，比 oracle 头寸捕获率。
用法：python run_key_improve.py [--smoke]
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
from run_adaptive import project_class_evidence_rt, K

CACHE = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache')
CACHE2 = Path(r'E:/Hermes/paper_tracking/sensitivity/pred_cache2')
ANNOT = 64


def macro_ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def wacc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def margin(logits):
    s = np.sort(logits, axis=1)
    return s[:, -1] - s[:, -2]


def entropy(logits):
    p = softmax(logits)
    return -(p * np.log(p + 1e-12)).sum(1)


KEYS = ['s2', 'logratio', 'postdiff', 'margin_diff', 'g_lr_conf', 'g_lr_feat']


def run(smoke=False):
    rows = []
    for p in sorted(x.name[:-4] for x in CACHE.glob('*.npz')):
        d = np.load(CACHE / f'{p}.npz', allow_pickle=False)
        if 'skipped' in d.files:
            continue
        d2 = np.load(CACHE2 / f'{p}.npz', allow_pickle=False)
        Xs, Z = d2['Xs'], d2['Z']
        feats = d['feats']; y_ids = d['y_ids']; ports_all = d['ports_all']
        qpos = d['qpos']; spos = d['spos']
        base = d['base_logits'].argmax(1); expert = d['expert_logits'].argmax(1)
        b_logits = d['base_logits']; e_logits = d['expert_logits']
        query_n = len(qpos); source_n = min(query_n, len(spos))
        rng_s = np.random.default_rng(1000)
        strat = {c: rng_s.choice(spos[y_ids[spos] == c],
                                 max(1, int(round(source_n * (y_ids[spos] == c).sum() / len(spos)))), replace=False)
                 for c in np.unique(y_ids[spos])}
        source_pos = np.concatenate(list(strat.values()))
        yq = y_ids[qpos]; bq = base[qpos]; eq = expert[qpos]
        bc = (bq == yq); ec = (eq == yq)
        D = (bq != eq)

        # 键 1/2/3（类证据）
        ann = np.random.default_rng(0).choice(qpos, ANNOT, replace=False)
        ay = y_ids[ann]
        observed = np.bincount(ay, minlength=K) > 0
        sm = fit_source(feats[source_pos], y_ids[source_pos], ports_all[source_pos])
        p1 = shared_cov_predict(sm, feats[ann], ay, feats[qpos])
        f1 = class_mass(p1['loglikelihood'], p1['counts'])
        ratio2 = project_class_evidence_rt(p1['loglikelihood'], f1['prior'])
        s2 = robust_scores(ratio2, bq, eq, observed)
        logratio = np.log(np.clip(ratio2[np.arange(query_n), eq] / np.clip(ratio2[np.arange(query_n), bq], 1e-9, None), 1e-9, None))
        postdiff = f1['posterior'][np.arange(query_n), eq] - f1['posterior'][np.arange(query_n), bq]

        # 键 4（margin 差）
        margin_diff = margin(e_logits[qpos]) - margin(b_logits[qpos])

        # 键 5/6（源侧 g 预测器）
        def conf_feats(lg_b, lg_e):
            return np.stack([margin(lg_e), margin(lg_b), entropy(lg_e), entropy(lg_b)], 1)
        # 源侧分歧样本
        sb = base[spos]; se = expert[spos]
        Ds = (sb != se)
        sbc = (sb == y_ids[spos]); sec = (se == y_ids[spos])
        g_src = (sec.astype(int) > sbc.astype(int))
        X_conf_s = conf_feats(b_logits[spos][Ds], e_logits[spos][Ds])
        y_s = g_src[Ds].astype(int)
        X_conf_q = conf_feats(b_logits[qpos], e_logits[qpos])
        g_conf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(X_conf_s, y_s)
        g_conf_score = g_conf.decision_function(X_conf_q)

        X_feat_s = np.hstack([conf_feats(b_logits[spos][Ds], e_logits[spos][Ds]),
                              Xs[spos][Ds], Z[spos][Ds]])
        X_feat_q = np.hstack([X_conf_q, Xs[qpos], Z[qpos]])
        g_feat = LogisticRegression(max_iter=2000, class_weight='balanced').fit(X_feat_s, y_s)
        g_feat_score = g_feat.decision_function(X_feat_q)

        keys = {'s2': s2, 'logratio': logratio, 'postdiff': postdiff,
                'margin_diff': margin_diff, 'g_lr_conf': g_conf_score, 'g_lr_feat': g_feat_score}

        # oracle
        oracle_pred = np.where(D, np.where(ec, eq, bq), bq)
        base_ba = macro_ba(yq, bq); orc_ba = macro_ba(yq, oracle_pred)

        row = {'port': p, 'base': base_ba, 'oracle': orc_ba}
        for k, sc in keys.items():
            bidir = np.where(D, np.where(sc > 0, eq, bq), bq)
            row[f'ba_{k}'] = macro_ba(yq, bidir)
            row[f'acc_{k}'] = wacc(yq, bidir)
            row[f'cap_{k}'] = ((row[f'ba_{k}'] - base_ba) / max(orc_ba - base_ba, 1e-9))
        rows.append(row)
        print(f'{p}: base={base_ba:.3f} orc={orc_ba:.3f} ' +
              ' '.join(f'{k}={row[f"cap_{k}"]:.2f}' for k in KEYS), flush=True)
        if smoke:
            break

    json.dump(rows, open('key_improve.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for k in KEYS:
        ba = np.mean([r[f'ba_{k}'] for r in rows]); acc = np.mean([r[f'acc_{k}'] for r in rows])
        cap = np.mean([r[f'cap_{k}'] for r in rows])
        print(f'{k:12s} BA={ba:.4f} ACC={acc:.4f} cap={cap:.3f}')
    print('base', np.mean([r['base'] for r in rows]), 'oracle', np.mean([r['oracle'] for r in rows]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=a.smoke)
