"""Matched, fixed-parameter OOF reliability diagnostic. No target-query tuning."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'frozen_modules'))
from bdh_pixel_loss_controls.conditional_likelihood import fit_source, class_mass, robust_scores
from bdh_pixel_loss_controls.shared_covariance_likelihood import predict as covariance_predict
from capacity_retrieval import retrieve
from report_bdh_pixel_handoff import source_weights

K = 8
ANNOT = 64
SEEDS = (0, 1, 2, 3, 4)
# Preregistered constants. Changing these after inspecting target results is a new exploratory run.
SOURCE_RIDGE_LAMBDA = .01  # weighted-mean squared loss + lambda ||theta||^2
SOURCE_LR_C = 1.
TARGET_PRIOR_STRENGTH = 8.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def confidence_features(base_logits, expert_logits):
    def parts(logits):
        x = np.asarray(logits, dtype=np.float64)
        require(x.ndim == 2 and x.shape[1] == K and np.isfinite(x).all(), 'invalid logits')
        order = np.sort(x, axis=1)
        p = np.exp(x - x.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(1)
        return order[:, -1] - order[:, -2], entropy
    bm, bh = parts(base_logits)
    em, eh = parts(expert_logits)
    return np.column_stack((em, bm, eh, bh))


def utility(labels, base, expert):
    return (np.asarray(expert) == labels).astype(float) - (np.asarray(base) == labels).astype(float)


def design(x, center, scale):
    return np.column_stack(((x - center) / scale, np.ones(len(x))))


def fit_reliability(base_logits, expert_logits, labels, ports):
    """Source counts are computed BEFORE disagreement/informative filtering."""
    x = confidence_features(base_logits, expert_logits)
    b, e = base_logits.argmax(1), expert_logits.argmax(1)
    delta = utility(labels, b, e)
    w_all = source_weights(np.asarray(labels), np.asarray(ports))
    center = np.average(x, axis=0, weights=w_all)
    scale = np.maximum(np.sqrt(np.average((x-center)**2, axis=0, weights=w_all)), 1e-6)
    X = design(x, center, scale)
    D = b != e
    require(D.any(), 'no source disagreements; reliability model cannot be fitted')
    wd = w_all[D] / w_all[D].sum()
    ridge = np.linalg.solve(X[D].T @ (wd[:, None] * X[D]) + SOURCE_RIDGE_LAMBDA * np.eye(X.shape[1]),
                            X[D].T @ (wd * delta[D]))
    informative = D & (delta != 0)
    require(np.unique(delta[informative]).size == 2, 'source informative choices lack both signs')
    # Proper binary baseline discards zero-utility rows. No class_weight='balanced'.
    lr = LogisticRegression(C=SOURCE_LR_C, max_iter=2000, solver='lbfgs', tol=1e-9)
    lr.fit(X[informative, :-1], (delta[informative] > 0).astype(int),
           sample_weight=len(labels) * w_all[informative])
    require(lr.n_iter_.max() < lr.max_iter, 'source LR did not converge')
    lr_theta = np.r_[lr.coef_[0], lr.intercept_[0]]
    return dict(center=center, scale=scale, ridge=ridge, lr=lr_theta,
                source_disagreements=int(D.sum()), source_informative=int(informative.sum()),
                source_zero_utility=int((D & (delta == 0)).sum()))


def adapt_reliability(model, support_features, labels, base, expert):
    """Target labels and class frequencies are restricted to the 64 support rows."""
    X = design(support_features, model['center'], model['scale'])
    delta = utility(labels, base, expert)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    # Before filtering: support-balanced weights sum to the support budget (64).
    w = len(labels) / (len(counts) * counts[inverse])
    D = base != expert
    if D.any():
        ridge = np.linalg.solve(X[D].T @ (w[D, None] * X[D]) + TARGET_PRIOR_STRENGTH * np.eye(X.shape[1]),
                                X[D].T @ (w[D] * delta[D]) + TARGET_PRIOR_STRENGTH * model['ridge'])
    else:
        ridge = model['ridge'].copy()
    informative = D & (delta != 0)
    lr_theta = model['lr'].copy()
    if informative.any():
        Xi, wi, yi = X[informative], w[informative], (delta[informative] > 0).astype(float)
        anchor = model['lr']
        def objective(theta):
            score = Xi @ theta
            loss = np.sum(wi * (np.logaddexp(0., score) - yi * score))
            difference = theta - anchor
            gradient = Xi.T @ (wi * (expit(score) - yi)) + TARGET_PRIOR_STRENGTH * difference
            return loss + .5 * TARGET_PRIOR_STRENGTH * (difference @ difference), gradient
        solved = minimize(objective, anchor, jac=True, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-7})
        require(solved.success, 'target LR adaptation failed: ' + str(solved.message))
        lr_theta = solved.x
    return dict(ridge=ridge, lr=lr_theta, support_disagreements=int(D.sum()),
                support_informative=int(informative.sum()))


def reliability_predictions(model, target_base, target_expert, support_local, support_labels):
    x = confidence_features(target_base, target_expert)
    b, e = target_base.argmax(1), target_expert.argmax(1)
    X = design(x, model['center'], model['scale'])
    adapted = adapt_reliability(model, x[support_local], support_labels, b[support_local], e[support_local])
    observed = np.bincount(support_labels, minlength=K) > 0
    eligible = (b != e) & observed[e]
    output = {}
    for name, theta in [('lr0', model['lr']), ('utility0', model['ridge']),
                        ('lr64', adapted['lr']), ('utility64', adapted['ridge'])]:
        output[name] = np.where(eligible & ((X @ theta) > 0), e, b)
    return output, {k:v for k,v in adapted.items() if k not in ('ridge','lr')}


def reference_predictions(d, annotation_rows):
    """Exact c82933b simple/s2 protocol; target query labels never enter fitting."""
    q, s = d['qpos'], d['spos']
    y, ports, feats = d['y_ids'], d['ports_all'], d['feats']
    n_source = min(len(q), len(s))
    rng = np.random.default_rng(1000)
    by_class = [rng.choice(s[y[s] == c], max(1, int(round(n_source * (y[s] == c).sum()/len(s)))),
                           replace=False) for c in np.unique(y[s])]
    chosen = np.concatenate(by_class)
    source = fit_source(feats[chosen], y[chosen], ports[chosen])
    likelihood = covariance_predict(source, feats[annotation_rows], y[annotation_rows], feats[q])
    ll = likelihood['loglikelihood']
    b = d['base_logits'][q].argmax(1)
    e = d['expert_logits'][q].argmax(1)
    observed = likelihood['counts'] > 0
    simple = np.where((b != e) & observed[e] & (ll[np.arange(len(q)),e] > ll[np.arange(len(q)),b]), e, b)
    mass = class_mass(ll, likelihood['counts'])
    try:
        retrieved = retrieve(-ll, mass['prior'], 'capacity')
    except ValueError:
        # Same fallback as pinned run_adaptive.project_class_evidence_rt.
        import ot
        shifted = -ll - (-ll).min(1, keepdims=True)
        coupling = ot.sinkhorn(np.full(len(ll), 1/len(ll)), mass['prior'], shifted,
                              reg=1., numItermax=50000, stopThr=1e-11)
        weights = len(ll) * coupling
        require(np.isfinite(weights).all() and
                np.abs(weights.mean(0)-mass['prior']).max()<=1e-8 and
                np.abs(weights.sum(1)-1).max()<=1e-8, 'fallback OT solver did not converge')
        retrieved = {'weights':weights}
    rho = retrieved['weights'] / mass['prior'][None]
    scores = robust_scores(rho, b, e, observed)
    s2 = np.where((b != e) & (scores > 0), e, b)
    return dict(base=b, fusion=e, simple=simple, s2=s2)


def load_oof(path, d, outer_cache_path, port, manifest_path=None):
    with np.load(path, allow_pickle=False) as a:
        required = {'source_rows','oof_base_logits','oof_expert_logits','source_labels','source_ports','metadata_json'}
        require(required <= set(a.files), 'OOF archive schema keys missing')
        out = {k: a[k] for k in required if k != 'metadata_json'}
        metadata = json.loads(str(a['metadata_json'].item()))
    require(metadata.get('schema') == 'oof-reliability-v1' and metadata.get('complete') is True,
            'partial/unrecognised OOF archive cannot produce a main result')
    require(metadata.get('outer_port') == port, 'OOF outer port mismatch')
    require(metadata.get('outer_cache_sha256') == sha256(outer_cache_path), 'OOF archive uses different outer cache')
    require(metadata.get('encoder_access') in ('independent','transductive-unlabeled'), 'missing upstream access declaration')
    require(metadata.get('outer_validation',{}).get('base_argmax_mismatches') == 0 and
            metadata.get('outer_validation',{}).get('expert_argmax_mismatches') == 0,
            'OOF archive has no successful frozen outer refit comparison')
    require(manifest_path is not None, '--manifest is required to independently verify OOF entity exclusions')
    from build_oof_predictions import load_manifest, outer_split, inner_split
    require(metadata.get('manifest_sha256') == sha256(manifest_path), 'OOF manifest fingerprint mismatch')
    manifest = load_manifest(manifest_path)
    require(np.array_equal(manifest['labels'],d['y_ids']) and
            np.array_equal(manifest['ports'],d['ports_all']), 'manifest/cache row alignment differs')
    expected_q, expected_s = outer_split(port,manifest)
    require(np.array_equal(expected_q,d['qpos']) and np.array_equal(expected_s,d['spos']),
            'outer cache split differs from manifest identity exclusion')
    rows = out['source_rows']
    require(np.array_equal(rows, d['spos']) and len(np.unique(rows)) == len(rows), 'OOF coverage/order mismatch')
    require(np.array_equal(out['source_labels'],d['y_ids'][rows]) and
            np.array_equal(out['source_ports'],d['ports_all'][rows]), 'OOF row labels/ports mismatch')
    require(not np.intersect1d(rows,d['qpos']).size and np.all(out['source_ports'] != port), 'OOF outer target overlap')
    for key in ['oof_base_logits','oof_expert_logits']:
        require(out[key].shape == (len(rows),K) and np.isfinite(out[key]).all(), 'invalid OOF predictions')
    covered = []
    for fold in metadata.get('folds',[]):
        train, held = np.asarray(fold['train_rows']), np.asarray(fold['heldout_rows'])
        expected_train, expected_held = inner_split(rows,fold['heldout_port'],d['qpos'],manifest)
        require(np.array_equal(train,expected_train) and np.array_equal(held,expected_held),
                'OOF fold differs from strict port/MMSI/product exclusion')
        require(not np.intersect1d(train, held).size and not np.intersect1d(train,d['qpos']).size,
                'OOF fold training overlap')
        require(np.all(d['ports_all'][train] != fold['heldout_port']) and
                np.all(d['ports_all'][held] == fold['heldout_port']), 'OOF port isolation failed')
        require(np.isin(train,d['spos']).all() and np.isin(held,rows).all(), 'OOF fold has extra rows')
        covered.extend(held.tolist())
    require(len(covered)==len(rows) and np.array_equal(np.sort(covered),np.sort(rows)), 'OOF fold provenance incomplete')
    out['metadata'] = metadata
    return out


def ba(labels, prediction):
    require(len(labels)>0, 'empty evaluation')
    return float(np.mean([np.mean(prediction[labels==c]==c) for c in np.unique(labels)]))


def score_predictions(predictions, labels, ev):
    base, expert = predictions['base'], predictions['fusion']
    delta = utility(labels,base,expert)
    classes, inverse, counts = np.unique(labels[ev],return_inverse=True,return_counts=True)
    w = np.zeros(len(labels))
    w[ev] = 1./(len(classes)*counts[inverse])
    oracle = np.where(delta>0,expert,base)
    predictions = dict(predictions,oracle=oracle)
    metrics = {}
    for name,p in predictions.items():
        selected = (base!=expert) & (p==expert)
        require(np.all((p==base)|(p==expert)), 'gate output is outside fixed model pair')
        recover = selected & (delta>0)
        harm = selected & (delta<0)
        metrics[name] = {'ba':ba(labels[ev],p[ev]), 'acc':float(np.mean(labels[ev]==p[ev])),
                         'recover_ba':float(w[recover].sum()), 'harm_ba':float(w[harm].sum()),
                         'oracle_gap_ba':ba(labels[ev],oracle[ev])-ba(labels[ev],p[ev])}
    return metrics


def paired_summary(rows):
    arms = list(rows[0]['seeds'][0]['metrics'])
    values = {a:np.array([np.mean([s['metrics'][a]['ba'] for s in r['seeds']]) for r in rows]) for a in arms}
    comparisons = {}
    candidates = [('s2','simple'),('simple','fusion')]
    for arm in arms:
        if arm.startswith('oof_'):
            candidates.append((arm,'simple'))
    for a,b in [('oof_utility64','oof_lr64'),('oof_utility64','oof_utility0'),
                ('oof_utility0','insample_utility0'),('oof_lr0','insample_lr0')]:
        if a in values and b in values:
            candidates.append((a,b))
    rng = np.random.default_rng(0)
    sampled = rng.integers(0,len(rows),size=(10000,len(rows)))
    for a,b in candidates:
        diff = values[a]-values[b]
        boot = diff[sampled].mean(1)
        comparisons[a+' minus '+b] = {'mean_delta_ba':float(diff.mean()),
            'positive_ports':int((diff>0).sum()), 'negative_ports':int((diff<0).sum()),
            'port_bootstrap_95ci':np.percentile(boot,[2.5,97.5]).tolist(),
            'wilcoxon_p':float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.}
    return {'n_ports':len(rows),'seeds_per_port':len(SEEDS),'means_ba':{a:float(v.mean()) for a,v in values.items()},
            'comparisons':comparisons,'statistical_unit':'port mean over five overlapping support seeds'}


def run(args):
    require(not args.out.exists() or not any(args.out.iterdir()),
            'output directory is not empty; use a new --out to avoid stale results')
    args.out.mkdir(parents=True,exist_ok=True)
    files = sorted(args.pred_cache.glob('*.npz'))
    require(bool(files),'no prediction caches')
    if args.ports:
        names=set(args.ports)
        files=[p for p in files if p.stem in names]
        require({p.stem for p in files}==names, 'requested port cache missing')
    rows=[]
    skipped=[]
    for path in files:
        with np.load(path,allow_pickle=False) as a:
            if 'skipped' in a.files:
                skipped.append(path.stem)
                continue
            d={k:a[k] for k in a.files}
        q,s=d['qpos'],d['spos']
        if len(q)<=ANNOT:
            skipped.append(path.stem)
            continue
        require(not np.intersect1d(q,s).size and np.all(d['ports_all'][q]==path.stem), 'invalid outer split')
        models={}
        if not args.reference_only:
            require(args.oof_cache is not None,'--oof-cache is required without --reference-only')
            oof=load_oof(args.oof_cache/path.name,d,path,path.stem,args.manifest)
            models['oof']=fit_reliability(oof['oof_base_logits'],oof['oof_expert_logits'],oof['source_labels'],oof['source_ports'])
            models['insample']=fit_reliability(d['base_logits'][s],d['expert_logits'][s],d['y_ids'][s],d['ports_all'][s])
        record={'port':path.stem,'outer_cache_sha256':sha256(path),'source_n':len(s),'target_n':len(q),'seeds':[]}
        if models:
            record['source_gate_diagnostics']={name:{k:v for k,v in m.items() if k.startswith('source_')} for name,m in models.items()}
            record['oof_encoder_access']=oof['metadata']['encoder_access']
        for seed in SEEDS:
            annotation=np.random.default_rng(seed).choice(q,ANNOT,replace=False)
            require(len(np.unique(annotation))==ANNOT,'support rows duplicated')
            ev=~np.isin(q,annotation)
            support_local=np.flatnonzero(~ev)
            predictions=reference_predictions(d,annotation)
            gate_diag={}
            for name,m in models.items():
                choices,diag=reliability_predictions(m,d['base_logits'][q],d['expert_logits'][q],support_local,d['y_ids'][q[support_local]])
                if name=='insample':
                    choices={a:p for a,p in choices.items() if a.endswith('0')}
                predictions.update({name+'_'+a:p for a,p in choices.items()})
                gate_diag[name]=diag
            metrics=score_predictions(predictions,d['y_ids'][q],ev)
            record['seeds'].append({'seed':seed,'annotation_rows':annotation.tolist(),'eval_n':int(ev.sum()),
                                    'observed_classes':np.unique(d['y_ids'][annotation]).tolist(),
                                    'support_diagnostics':gate_diag,'metrics':metrics})
        rows.append(record)
        (args.out/(path.stem+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(path.stem+': '+ ' '.join(a+'='+format(np.mean([r['metrics'][a]['ba'] for r in record['seeds']]),'.6f')
                                      for a in record['seeds'][0]['metrics']),flush=True)
    require(bool(rows),'no evaluable ports')
    summary=paired_summary(rows)
    summary.update(reference_only=bool(args.reference_only),skipped_ports=skipped,
                   protocol={'source_ridge_lambda':SOURCE_RIDGE_LAMBDA,'source_lr_C':SOURCE_LR_C,
                   'target_prior_strength':TARGET_PRIOR_STRENGTH,'query_tuning':False,
                   'features':'expert/base margins and entropies (4D)','support_budget':ANNOT,
                   'all_learned_gates_use_same_observed_expert_guard':True,
                   'status':'development diagnostic; previously inspected ports are not fresh blind confirmation'})
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pred-cache',type=Path,required=True)
    p.add_argument('--oof-cache',type=Path)
    p.add_argument('--manifest',type=Path,help='required for learned OOF evaluation and entity-provenance checks')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ports',nargs='+')
    p.add_argument('--reference-only',action='store_true',help='replay frozen references ONLY; never claim an OOF result')
    p.add_argument('--threads',type=int,default=1)
    a=p.parse_args()
    with threadpool_limits(limits=a.threads):
        run(a)


if __name__=='__main__':
    main()
