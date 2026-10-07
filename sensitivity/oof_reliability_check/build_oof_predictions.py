#!/usr/bin/env python3
"""Build genuine source-port OOF head predictions on supplied frozen embeddings.

This reproduces the PCA128/64 + balanced Ridge(alpha=1) head recipe, checks it
against each frozen outer cache, then refits the heads for every source port.
It does NOT retrain an encoder. Results are conditional on the provided frozen
embeddings. --encoder-access is an explicit upstream access declaration, not a
proof of encoder/scene legality. Unknown or SAR-supervised encoder access is
unsupported without a separate per-port encoder retraining interface.

Example:
  python build_oof_predictions.py --manifest manifest.csv \
    --vv-features pooled_features.npy --vh-features b2_ssl_feats_38091.npz \
    --pred-cache pred_cache --out oof_cache --threads 1 \
    --encoder-access transductive-unlabeled

--inner-ports is for partial smoke builds. Partial caches have complete=false
and must not be used as a main evaluation. Small inner ports are not skipped.
Run --self-test for a synthetic split and target-label independence check.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

SCHEMA = 'oof-reliability-v1'
RECIPE_VERSION = 'pca128-pca64-standardize-balanced-ridge1-v1'
K = 8
ACCESS_CHOICES = ('independent', 'transductive-unlabeled')


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_sha256(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def load_manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        names = set(reader.fieldnames or [])
        require({'port', 'class_id', 'mmsi'} <= names,
                'manifest must contain port, class_id, mmsi columns')
        require('product' in names or 'product_id' in names,
                'manifest must contain product or product_id for identity exclusion')
        rows = list(reader)
    require(rows, 'empty manifest')
    ports = np.asarray([r['port'] for r in rows], dtype=str)
    require(np.all(ports != ''), 'empty port in manifest')
    try:
        labels = np.asarray([int(r['class_id']) for r in rows], dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError('class_id must be an integer in 0..7') from exc
    require(np.all((labels >= 0) & (labels < K)), 'manifest class_id outside 0..7')
    require(np.array_equal(np.unique(labels), np.arange(K)),
            'manifest must use all eight aligned class ids 0..7')
    mmsi = np.asarray([r.get('mmsi') or '' for r in rows], dtype=str)
    product = np.asarray([r.get('product') or r.get('product_id') or '' for r in rows], dtype=str)
    return dict(labels=labels, ports=ports, mmsi=mmsi, product=product)


def load_features(path, key=None):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        data = np.load(path, mmap_mode='r', allow_pickle=False)
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            chosen = key
            if chosen is None:
                candidates = [k for k in ('F', 'features', 'feats') if k in archive.files]
                if len(candidates) == 1:
                    chosen = candidates[0]
                elif len(archive.files) == 1:
                    chosen = archive.files[0]
                else:
                    raise ValueError(f'{path}: cannot infer unique feature key; use a .npy VV array')
            require(chosen in archive.files, f'{path}: missing feature key {chosen!r}')
            data = archive[chosen]
    else:
        raise ValueError(f'{path}: expected .npy or .npz features')
    require(data.ndim == 2 and data.dtype.kind in 'fiu', f'{path}: features must be numeric N x D')
    data = np.asarray(data, dtype=np.float32)
    require(np.isfinite(data).all(), f'{path}: nonfinite frozen embeddings')
    return data


def checked_rows(value, n, name):
    arr = np.asarray(value)
    require(arr.ndim == 1 and arr.dtype.kind in 'iu', f'{name}: expected 1-D integer rows')
    arr = arr.astype(np.int64, copy=False)
    require(np.all((arr >= 0) & (arr < n)), f'{name}: out-of-bounds row')
    require(len(np.unique(arr)) == len(arr), f'{name}: duplicate row')
    return arr


def exclude_identities(candidate_rows, heldout_rows, manifest):
    candidates = np.asarray(candidate_rows, dtype=np.int64)
    keep = np.ones(len(candidates), dtype=bool)
    for key in ('mmsi', 'product'):
        values = manifest[key]
        held_ids = np.unique(values[heldout_rows])
        held_ids = held_ids[held_ids != '']
        keep &= ~np.isin(values[candidates], held_ids)
    return candidates[keep]


def assert_identity_disjoint(train_rows, heldout_rows, manifest, description):
    require(not np.intersect1d(train_rows, heldout_rows).size,
            f'{description}: direct row overlap')
    for key in ('mmsi', 'product'):
        a = np.unique(manifest[key][train_rows]); a = a[a != '']
        b = np.unique(manifest[key][heldout_rows]); b = b[b != '']
        require(not np.intersect1d(a, b).size, f'{description}: shared nonempty {key}')


def outer_split(port, manifest):
    query_rows = np.flatnonzero(manifest['ports'] == port)
    require(query_rows.size, f'unknown outer port {port!r}')
    candidates = np.flatnonzero(manifest['ports'] != port)
    source_rows = exclude_identities(candidates, query_rows, manifest)
    assert_identity_disjoint(source_rows, query_rows, manifest, f'outer {port}')
    return query_rows, source_rows


def inner_split(source_rows, inner_port, outer_rows, manifest):
    heldout = source_rows[manifest['ports'][source_rows] == inner_port]
    require(heldout.size, f'inner port {inner_port!r} has no outer-source rows')
    # Identity exclusion uses ALL rows of h, including h rows already excluded
    # by the outer split; heldout prediction still covers only outer-source rows.
    all_inner_rows = np.flatnonzero(manifest['ports'] == inner_port)
    candidates = source_rows[manifest['ports'][source_rows] != inner_port]
    train = exclude_identities(candidates, all_inner_rows, manifest)
    assert_identity_disjoint(train, all_inner_rows, manifest, f'inner {inner_port}')
    assert_identity_disjoint(train, outer_rows, manifest, f'inner {inner_port}, outer target')
    require(not np.intersect1d(heldout, outer_rows).size, 'outer-target rows in OOF heldout')
    return train, heldout


def fit_predict_heads(vv, vh, labels, train_rows, predict_rows):
    """Frozen recipe, returning predictions only; no model remains resident."""
    require(len(train_rows) >= 128 and vv.shape[1] >= 128 and vh.shape[1] >= 64,
            'too few training rows/features for frozen PCA128/64 recipe')
    require(np.array_equal(np.unique(labels[train_rows]), np.arange(K)),
            'head training fold lacks an aligned class: all ids 0..7 are required')
    full_prediction = (len(predict_rows) == len(labels)
                       and np.array_equal(predict_rows, np.arange(len(labels))))

    pca = PCA(n_components=128, random_state=0).fit(vv[train_rows])
    x_pred = pca.transform(vv[predict_rows])
    x_train = x_pred[train_rows] if full_prediction else pca.transform(vv[train_rows])
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-6
    xs_train = (x_train - mu) / sd
    xs_pred = (x_pred - mu) / sd
    base_model = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(xs_train, labels[train_rows])
    require(np.array_equal(base_model.classes_, np.arange(K)), 'base class-column misalignment')
    base_logits = base_model.decision_function(xs_pred)
    del base_model, pca, x_pred, x_train

    pca2 = PCA(n_components=64, random_state=0).fit(vh[train_rows])
    z_pred = pca2.transform(vh[predict_rows])
    z_train = z_pred[train_rows] if full_prediction else pca2.transform(vh[train_rows])
    m_train = np.c_[xs_train, z_train]
    m_pred = np.c_[xs_pred, z_pred]
    mu2, sd2 = m_train.mean(0), m_train.std(0) + 1e-6
    expert_model = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(
        (m_train - mu2) / sd2, labels[train_rows])
    require(np.array_equal(expert_model.classes_, np.arange(K)), 'expert class-column misalignment')
    expert_logits = expert_model.decision_function((m_pred - mu2) / sd2)
    del expert_model, pca2, z_pred, z_train, xs_train, xs_pred, m_train, m_pred
    require(base_logits.shape == expert_logits.shape == (len(predict_rows), K),
            'unexpected head decision-function shape')
    require(np.isfinite(base_logits).all() and np.isfinite(expert_logits).all(),
            'nonfinite head logits')
    return np.asarray(base_logits, dtype=np.float32), np.asarray(expert_logits, dtype=np.float32)


def read_outer_cache(path, port, manifest):
    n = len(manifest['labels'])
    with np.load(path, allow_pickle=False) as archive:
        required = {'qpos', 'spos', 'y_ids', 'ports_all', 'base_logits', 'expert_logits'}
        require(required <= set(archive.files),
                f'{path}: missing complete outer cache keys; a skipped-small-port cache cannot be outer target')
        q = checked_rows(archive['qpos'], n, 'outer qpos')
        s = checked_rows(archive['spos'], n, 'outer spos')
        require(np.array_equal(archive['y_ids'], manifest['labels']),
                f'{path}: cached labels do not align with manifest rows')
        require(np.array_equal(archive['ports_all'].astype(str), manifest['ports']),
                f'{path}: cached ports do not align with manifest rows')
        base = np.asarray(archive['base_logits'], dtype=np.float32)
        expert = np.asarray(archive['expert_logits'], dtype=np.float32)
    expected_q, expected_s = outer_split(port, manifest)
    require(np.array_equal(q, expected_q), f'{path}: qpos is not full sorted outer-port rows')
    require(np.array_equal(s, expected_s), f'{path}: spos differs from frozen outer identity-exclusion recipe')
    require(base.shape == expert.shape == (n, K), f'{path}: expected N x 8 outer logits')
    require(np.isfinite(base).all() and np.isfinite(expert).all(), f'{path}: nonfinite cached logits')
    return q, s, base, expert


def validate_outer_heads(vv, vh, manifest, source_rows, cached_base, cached_expert, args):
    started = time.perf_counter()
    base, expert = fit_predict_heads(vv, vh, manifest['labels'], source_rows,
                                     np.arange(len(manifest['labels'])))
    result = {'seconds': time.perf_counter() - started,
              'atol': args.logit_atol, 'rtol': args.logit_rtol, 'compared_rows': len(base)}
    for name, actual, reference in (('base', base, cached_base), ('expert', expert, cached_expert)):
        result[name + '_max_abs_logit_error'] = float(np.max(np.abs(actual - reference)))
        mismatches = int(np.count_nonzero(actual.argmax(1) != reference.argmax(1)))
        result[name + '_argmax_mismatches'] = mismatches
        require(mismatches == 0,
                f'outer {name} refit has {mismatches} prediction mismatches; check frozen embeddings, '
                'PCA/scikit-learn version, row alignment and recipe before building OOF')
        require(np.allclose(actual, reference, atol=args.logit_atol, rtol=args.logit_rtol),
                f'outer {name} logits exceed tolerance (max error '
                f'{result[name + "_max_abs_logit_error"]:.6g}); do not silently use a different outer recipe')
    return result


def atomic_save(path, source_rows, base, expert, manifest, metadata):
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, source_rows=source_rows,
                                oof_base_logits=base, oof_expert_logits=expert,
                                source_labels=manifest['labels'][source_rows],
                                source_ports=manifest['ports'][source_rows],
                                metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_saved(path, fingerprint, source_rows, outer_rows, manifest, outer_port):
    with np.load(path, allow_pickle=False) as archive:
        require({'source_rows', 'oof_base_logits', 'oof_expert_logits', 'source_labels',
                 'source_ports', 'metadata_json'} <= set(archive.files), 'invalid OOF cache schema')
        raw = archive['metadata_json']
        require(raw.ndim == 0 and raw.dtype.kind in 'US', 'metadata_json must be a Unicode/string scalar')
        metadata = json.loads(str(raw.item()))
        require(metadata.get('schema') == SCHEMA, 'OOF schema differs; rebuild explicitly')
        require(metadata.get('fingerprint') == fingerprint,
                f'{path}: resume fingerprint mismatch; choose a fresh output directory')
        require(metadata.get('outer_port') == outer_port, 'OOF outer port mismatch')
        require(np.array_equal(archive['source_rows'], source_rows), 'OOF source-row alignment mismatch')
        require(np.array_equal(archive['source_labels'], manifest['labels'][source_rows]),
                'OOF source labels mismatch')
        require(np.array_equal(archive['source_ports'].astype(str), manifest['ports'][source_rows]),
                'OOF source ports mismatch')
        base, expert = archive['oof_base_logits'].copy(), archive['oof_expert_logits'].copy()
    require(base.shape == expert.shape == (len(source_rows), K), 'invalid saved OOF logits shape')
    coverage = np.zeros(len(source_rows), dtype=bool)
    done = set()
    for fold in metadata.get('folds', []):
        h = fold['heldout_port']
        require(h not in done, 'duplicate inner fold in OOF metadata')
        expected_train, expected_held = inner_split(source_rows, h, outer_rows, manifest)
        require(np.array_equal(np.asarray(fold['train_rows'], dtype=np.int64), expected_train),
                f'saved fold {h}: training rows differ')
        require(np.array_equal(np.asarray(fold['heldout_rows'], dtype=np.int64), expected_held),
                f'saved fold {h}: heldout rows differ')
        positions = np.searchsorted(source_rows, expected_held)
        require(not coverage[positions].any(), 'duplicate OOF heldout coverage')
        require(np.isfinite(base[positions]).all() and np.isfinite(expert[positions]).all(),
                f'saved fold {h}: nonfinite OOF logits')
        coverage[positions] = True
        done.add(h)
    require(isinstance(metadata.get('complete'), bool), 'metadata complete must be boolean')
    require(metadata['complete'] == bool(coverage.all()), 'complete flag does not match fold coverage')
    require(np.isnan(base[~coverage]).all() and np.isnan(expert[~coverage]).all(),
            'uncovered rows must be NaN and explicitly partial')
    return metadata, base, expert, coverage, done


def flatten_ports(values):
    if values is None:
        return None
    result = []
    for value in values:
        result.extend(p.strip() for p in value.split(',') if p.strip())
    require(result and len(set(result)) == len(result), 'empty or duplicate port selection')
    return result


def build(args):
    require(args.encoder_access in ACCESS_CHOICES,
            '--encoder-access is required: declare independent or transductive-unlabeled. '
            'Unknown/SAR-supervised encoder access cannot be validated by this frozen-embedding builder.')
    inputs = ('manifest', 'vv_features', 'vh_features', 'pred_cache', 'out')
    for name in inputs:
        require(getattr(args, name) is not None, f'--{name.replace("_", "-")} is required')
    paths = {name: Path(getattr(args, name)).expanduser().resolve() for name in inputs}
    require(paths['pred_cache'].is_dir(), 'prediction cache directory does not exist')
    paths['out'].mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(paths['manifest'])
    vv = load_features(paths['vv_features'])
    vh = load_features(paths['vh_features'], args.vh_key)
    require(len(vv) == len(vh) == len(manifest['labels']), 'feature/manifest row counts differ')
    common = dict(schema=SCHEMA, recipe_version=RECIPE_VERSION,
                  manifest_sha256=file_sha256(paths['manifest']),
                  vv_features_sha256=file_sha256(paths['vv_features']),
                  vh_features_sha256=file_sha256(paths['vh_features']),
                  input_paths={k: str(v) for k, v in paths.items() if k != 'out'},
                  vh_key=args.vh_key, encoder_access=args.encoder_access,
                  encoder_access_is_user_declaration=True,
                  conditional_on_provided_frozen_embeddings=True,
                  upstream_ssl_and_scene_legality_verified=False,
                  upstream_limitations=('No encoder retraining is performed. Encoder access and scene/ship '
                                        'provenance must be independently audited. Empty provided identities '
                                        'cannot be excluded; only nonempty manifest mmsi/product are checked.'),
                  classes=list(range(K)), vv_components=128, vh_components=64,
                  ridge_alpha=1.0, class_weight='balanced', scaler_epsilon=1e-6,
                  inner_identity_exclusion='all global manifest rows of heldout port',
                  pca_random_state=0, pca_svd_solver='auto',
                  numpy_version=np.__version__, scipy_version=scipy.__version__,
                  sklearn_version=sklearn.__version__, python_version=platform.python_version(),
                  threads=args.threads, logit_atol=args.logit_atol, logit_rtol=args.logit_rtol,
                  identity_missing_counts={k: int(np.sum(manifest[k] == '')) for k in ('mmsi', 'product')})
    selected_ports = flatten_ports(args.ports)
    if selected_ports is None:
        selected_ports = []
        for cache in sorted(paths['pred_cache'].glob('*.npz')):
            with np.load(cache, allow_pickle=False) as archive:
                if 'qpos' not in archive.files or 'spos' not in archive.files:
                    print(f'skip incomplete outer cache {cache.name}; small ports remain eligible as inner folds', flush=True)
                    continue
            selected_ports.append(cache.stem)
    require(selected_ports, 'no complete outer prediction caches selected')
    inner_selection = flatten_ports(args.inner_ports)
    with threadpool_limits(limits=args.threads):
        for port in selected_ports:
            require('/' not in port and '\\' not in port, 'port names may not contain path separators')
            outer_cache = paths['pred_cache'] / f'{port}.npz'
            require(outer_cache.is_file(), f'missing outer cache: {outer_cache}')
            outer_rows, source_rows, cached_base, cached_expert = read_outer_cache(outer_cache, port, manifest)
            available = sorted(np.unique(manifest['ports'][source_rows]).tolist())
            desired = available if inner_selection is None else inner_selection
            require(set(desired) <= set(available), f'{port}: requested inner port outside outer source set')
            fingerprint_inputs = dict(common, outer_port=port, outer_cache_sha256=file_sha256(outer_cache))
            fingerprint = object_sha256(fingerprint_inputs)
            destination = paths['out'] / f'{port}.npz'
            if destination.exists():
                metadata, base, expert, covered, done = validate_saved(
                    destination, fingerprint, source_rows, outer_rows, manifest, port)
                print(f'{port}: verified resume {int(covered.sum())}/{len(source_rows)} source rows', flush=True)
                # A validated checkpoint already carries its successful outer refit comparison.
                require(metadata.get('outer_validation', {}).get('base_argmax_mismatches') == 0
                        and metadata.get('outer_validation', {}).get('expert_argmax_mismatches') == 0,
                        'resume cache lacks successful outer recipe verification')
            else:
                print(f'{port}: validating frozen outer recipe before OOF', flush=True)
                validation = validate_outer_heads(vv, vh, manifest, source_rows, cached_base, cached_expert, args)
                metadata = dict(fingerprint_inputs, fingerprint=fingerprint, outer_validation=validation,
                                complete=False, source_n=len(source_rows), outer_query_n=len(outer_rows),
                                source_rows_order='exact outer cache spos order', available_inner_ports=available,
                                folds=[], coverage_rows=0)
                base = np.full((len(source_rows), K), np.nan, dtype=np.float32)
                expert = np.full_like(base, np.nan)
                covered = np.zeros(len(source_rows), dtype=bool)
                done = set()
            del cached_base, cached_expert
            metadata['requested_inner_ports'] = desired
            metadata['partial_smoke_requested'] = inner_selection is not None
            for h in desired:
                if h in done:
                    continue
                started = time.perf_counter()
                train, heldout = inner_split(source_rows, h, outer_rows, manifest)
                # This includes n<=64 ports: OOF training does not require inner query evaluation.
                b, e = fit_predict_heads(vv, vh, manifest['labels'], train, heldout)
                positions = np.searchsorted(source_rows, heldout)
                require(np.array_equal(source_rows[positions], heldout), 'OOF source index mapping failed')
                require(not covered[positions].any(), 'attempted duplicate OOF assignment')
                base[positions], expert[positions] = b, e
                del b, e
                covered[positions] = True
                done.add(h)
                metadata['folds'].append(dict(heldout_port=h, train_rows=train.tolist(),
                                              heldout_rows=heldout.tolist(), train_n=len(train),
                                              heldout_n=len(heldout), seconds=time.perf_counter() - started,
                                              training_class_counts=np.bincount(manifest['labels'][train], minlength=K).tolist()))
                metadata['coverage_rows'] = int(covered.sum())
                metadata['complete'] = bool(covered.all())
                require(not np.intersect1d(source_rows, outer_rows).size, 'outer-target rows in OOF cache')
                atomic_save(destination, source_rows, base, expert, manifest, metadata)
                print(f'{port} / {h}: train={len(train)}, heldout={len(heldout)}, '
                      f'{metadata["folds"][-1]["seconds"]:.1f}s, coverage='
                      f'{metadata["coverage_rows"]}/{len(source_rows)}', flush=True)
            # Save even a zero-new-fold resumed smoke run, retaining its complete/partial truth.
            metadata['complete'] = bool(covered.all())
            metadata['coverage_rows'] = int(covered.sum())
            if metadata['complete']:
                require(np.isfinite(base).all() and np.isfinite(expert).all(), 'complete OOF cache is nonfinite')
            atomic_save(destination, source_rows, base, expert, manifest, metadata)
            print(f'{port}: {"COMPLETE" if metadata["complete"] else "PARTIAL — not a main-result cache"} '
                  f'-> {destination}', flush=True)


def self_test():
    """Synthetic only; never writes or claims any real SAR OOF run."""
    rng = np.random.default_rng(18)
    labels = np.tile(np.repeat(np.arange(K), 18), 4)
    ports = np.repeat(np.asarray(['target', 'inner', 'source_a', 'source_b']), K * 18)
    n = len(labels)
    manifest = dict(labels=labels, ports=ports,
                    mmsi=np.asarray([f'ship-{i}' for i in range(n)]),
                    product=np.asarray([f'product-{i}' for i in range(n)]))
    # Deliberate identity links must also be excluded across ports.
    manifest['mmsi'][300] = manifest['mmsi'][0]
    manifest['product'][301] = manifest['product'][144]
    manifest['mmsi'][145] = manifest['mmsi'][0]
    manifest['product'][302] = manifest['product'][145]
    vv = rng.normal(size=(n, 160)).astype(np.float32)
    vh = rng.normal(size=(n, 80)).astype(np.float32)
    outer, source = outer_split('target', manifest)
    require(300 not in source, 'self-test outer identity exclusion failed')
    train, held = inner_split(source, 'inner', outer, manifest)
    require(301 not in train and 301 in source, 'self-test inner identity exclusion failed')
    require(145 not in source and 302 in source and 302 not in train,
            'self-test identity of outer-excluded inner row was not excluded')
    require(len(held) == 143, 'self-test inner coverage differs')
    changed = labels.copy(); changed[outer] = (changed[outer] + 3) % K
    with threadpool_limits(limits=1):
        before = fit_predict_heads(vv, vh, labels, train, held)
        after = fit_predict_heads(vv, vh, changed, train, held)
    require(all(np.array_equal(a, b) for a, b in zip(before, after)),
            'target label changes affected OOF head predictions')
    print(json.dumps(dict(self_test='passed', real_sar_oof_run=False,
                          identity_exclusion=True, target_label_independence=True,
                          synthetic_train_n=len(train), synthetic_heldout_n=len(held)), indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest')
    parser.add_argument('--vv-features')
    parser.add_argument('--vh-features')
    parser.add_argument('--vh-key', default='F')
    parser.add_argument('--pred-cache')
    parser.add_argument('--out')
    parser.add_argument('--ports', nargs='+', help='outer ports, quoted names or comma-separated')
    parser.add_argument('--inner-ports', nargs='+', help='partial smoke only; include small ports when testing')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--encoder-access', choices=ACCESS_CHOICES,
                        help='required upstream declaration; unknown/SAR-supervised encoders unsupported')
    parser.add_argument('--logit-atol', type=float, default=1e-3)
    parser.add_argument('--logit-rtol', type=float, default=1e-3)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    require(args.threads >= 1, '--threads must be positive')
    require(args.logit_atol >= 0 and args.logit_rtol >= 0, 'logit tolerances must be nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test()
    else:
        build(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
