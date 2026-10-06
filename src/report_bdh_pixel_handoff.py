"""Independent read-only replay helpers for budgeted pixel-expert handoff.

Parent predictions and completed roles are the input boundary. This module does
not import the pixel trainer, load torch checkpoints, or recompute training roles.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ('bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
           'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply')
ARMS = ('ce', 'hard', 'repair', 'repair_raw', 'ce_quality', 'repair_quality',
        'repair_shuffle_global', 'repair_shuffle_within', 'dual_b', 'dual_ab')
GATES = ('pair', 'expert_margin', 'base_entropy')
BUDGETS = (.05, .1, .2)
EPSILON = 1e-12
PRIMARY = 'repair_quality minus repair / pair / B05 equal-port BA'
KEY = ['seed', 'fold', 'port', 'arm', 'gate', 'budget']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def align_logits(archive, rows):
    """Align by global manifest indexes, never by archive position or role size."""
    indexes, logits, rows = np.asarray(archive['indexes']), np.asarray(archive['logits']), np.asarray(rows)
    require(indexes.ndim == rows.ndim == 1 and np.issubdtype(indexes.dtype, np.integer)
            and np.issubdtype(rows.dtype, np.integer), 'role/archive indexes must be integer vectors')
    require(len(np.unique(indexes)) == len(indexes) and len(np.unique(rows)) == len(rows), 'duplicate indexes')
    require(logits.shape == (len(indexes), len(CLASSES)) and np.isfinite(logits).all(), 'invalid parent logits')
    order = np.argsort(indexes)
    position = np.searchsorted(indexes[order], rows)
    require(np.all(position < len(indexes)), 'missing role indexes')
    require(np.array_equal(indexes[order[position]], rows), 'missing role indexes')
    return np.asarray(logits[order[position]], dtype=np.float64)


def source_weights(labels, ports):
    """Equal mass per source port, then per class present in that entire port."""
    labels, ports = np.asarray(labels), np.asarray(ports)
    require(labels.ndim == 1 and ports.shape == labels.shape and len(labels) > 0, 'invalid source rows')
    require(np.issubdtype(labels.dtype, np.integer) and np.all((labels >= 0) & (labels < len(CLASSES))),
            'invalid source classes')
    result = np.empty(len(labels), dtype=np.float64)
    domains = np.unique(ports)
    for port in domains:
        mask = ports == port
        classes, inverse, counts = np.unique(labels[mask], return_inverse=True, return_counts=True)
        result[mask] = 1. / (len(domains) * len(classes) * counts[inverse])
    return result


def pair_statistics(labels, base, expert, ports):
    weights = source_weights(labels, ports)
    labels, base, expert = map(np.asarray, (labels, base, expert))
    require(base.shape == expert.shape == labels.shape, 'source prediction alignment differs')
    require(all(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a < len(CLASSES)))
                for a in (base, expert)), 'invalid source predictions')
    gain = (expert == labels).astype(float) - (base == labels).astype(float)
    bins = len(CLASSES) * base + expert
    shape = (len(CLASSES), len(CLASSES))
    numerator = np.bincount(bins, weights=weights * gain, minlength=len(CLASSES) ** 2).reshape(shape)
    denominator = np.bincount(bins, weights=weights, minlength=len(CLASSES) ** 2).reshape(shape)
    support = np.bincount(bins, minlength=len(CLASSES) ** 2).reshape(shape)
    score = np.divide(numerator, denominator, out=np.zeros(shape), where=denominator > 0)
    return dict(score=score, numerator=numerator, denominator=denominator, support=support, weights=weights)


def probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    require(logits.ndim == 2 and logits.shape[1] == len(CLASSES) and np.isfinite(logits).all(), 'invalid logits')
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def gate_scores(base_logits, expert_logits, pair_table):
    pb, pe = probabilities(base_logits), probabilities(expert_logits)
    require(pb.shape == pe.shape, 'base/expert rows differ')
    table = np.asarray(pair_table)
    require(table.shape == (len(CLASSES), len(CLASSES)) and np.isfinite(table).all(), 'invalid pair table')
    base, expert = np.argmax(base_logits, axis=1), np.argmax(expert_logits, axis=1)
    rows = np.arange(len(base))
    return dict(pair=table[base, expert], expert_margin=pe[rows, expert] - pe[rows, base],
                base_entropy=-np.sum(pb * np.log(np.maximum(pb, 1e-300)), axis=1))


def tie_priority(seed, fold, n):
    permutation = np.random.default_rng(np.random.SeedSequence([seed, fold, 61801])).permutation(n)
    return np.argsort(permutation)


def budget_policy(base, expert, score, budget, priority):
    base, expert, score, priority = map(np.asarray, (base, expert, score, priority))
    n = len(base)
    require(n > 0 and base.ndim == 1 and expert.shape == score.shape == priority.shape == base.shape,
            'policy row alignment differs')
    require(np.isfinite(score).all(), 'nonfinite gate score')
    require(np.array_equal(np.sort(priority), np.arange(n)), 'tie priority must be a row permutation')
    require(budget in BUDGETS, 'unexpected budget')
    eligible = (base != expert) & (score > EPSILON)
    cap = min(n, max(1, round(budget * n)))
    order = np.lexsort((priority, -score))
    chosen = order[eligible[order]][:cap]
    selected = np.zeros(n, dtype=bool)
    selected[chosen] = True
    return dict(selected=selected, prediction=np.where(selected, expert, base), cap=cap,
                positive_eligible=int(eligible.sum()))


def oracle_policy(labels, base, expert, budget, priority):
    """Nondeployable BA upper bound: only repairs, rare-class contribution first."""
    labels, base, expert = map(np.asarray, (labels, base, expert))
    require(labels.shape == base.shape == expert.shape, 'oracle label alignment differs')
    counts = np.bincount(labels, minlength=len(CLASSES))
    score = ((expert == labels) & (base != labels)) / (np.count_nonzero(counts) * counts[labels])
    return budget_policy(base, expert, score, budget, priority)


def classification_metrics(labels, prediction):
    labels, prediction = map(np.asarray, (labels, prediction))
    support = np.bincount(labels, minlength=len(CLASSES))
    correct = np.bincount(labels[labels == prediction], minlength=len(CLASSES))
    recall = np.divide(correct, support, out=np.full(len(CLASSES), np.nan), where=support > 0)
    return dict(ba=float(recall[support > 0].mean()) if np.any(support) else float('nan'),
                accuracy=float(np.mean(labels == prediction)) if len(labels) else float('nan'),
                support=support, correct=correct, recall=recall)


def policy_readout(labels, base, expert, policy):
    labels, base, expert = map(np.asarray, (labels, base, expert))
    prediction, selected = policy['prediction'], policy['selected']
    require(labels.shape == base.shape == expert.shape == prediction.shape == selected.shape, 'readout rows differ')
    require(selected.dtype == np.bool_ and np.array_equal(prediction, np.where(selected, expert, base)),
            'not a two-model handoff')
    require(not np.any(selected & (base == expert)), 'agreement rows cannot be selected')
    repairs, harms = (base != labels) & (prediction == labels), (base == labels) & (prediction != labels)
    before, after = classification_metrics(labels, base), classification_metrics(labels, prediction)
    original_expert = classification_metrics(labels, expert)
    agreement = base == expert
    metrics = dict(n=len(labels), ba=after['ba'], base_ba=before['ba'], expert_ba=original_expert['ba'],
        accuracy=after['accuracy'], base_accuracy=before['accuracy'], expert_accuracy=original_expert['accuracy'],
        ba_gain=after['ba'] - before['ba'], selected_n=int(selected.sum()), changed_n=int(np.sum(prediction != base)),
        selected_fraction=float(selected.mean()), repairs=int(repairs.sum()), harms=int(harms.sum()),
        agreement_errors=int(np.sum(agreement & (base != labels))),
        agreement_repairs=int(np.sum(agreement & repairs)), agreement_harms=int(np.sum(agreement & harms)))
    classes = []
    for c, name in enumerate(CLASSES):
        mask = labels == c
        classes.append(dict(class_name=name, support=int(after['support'][c]), correct=int(after['correct'][c]),
            base_correct=int(before['correct'][c]), recall=float(after['recall'][c]), base_recall=float(before['recall'][c]),
            repairs=int(np.sum(repairs & mask)), harms=int(np.sum(harms & mask))))
    groups = []
    for name, mask in [('agreement', agreement), ('disagreement', ~agreement)]:
        groups.append(dict(group=name, n=int(mask.sum()), selected_n=int(np.sum(selected & mask)),
            correct=int(np.sum((prediction == labels) & mask)), base_correct=int(np.sum((base == labels) & mask)),
            base_wrong=int(np.sum((base != labels) & mask)), repairs=int(np.sum(repairs & mask)), harms=int(np.sum(harms & mask))))
    return dict(metrics=metrics, class_metrics=classes, group_metrics=groups)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def same(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, name + ': shape differs')
    if expected.dtype.kind in 'fc':
        require(np.allclose(actual.astype(float), expected, atol=1e-11, rtol=1e-11, equal_nan=True), name + ': values differ')
    else:
        require(np.array_equal(actual, expected), name + ': values differ')


def compare_csv(path, expected, keys):
    actual = pd.read_csv(path, keep_default_na=False, na_values=[''])
    require(set(actual.columns) == set(expected.columns), path.name + ': columns differ')
    require(not actual.duplicated(keys).any() and len(actual) == len(expected), path.name + ': row identities differ')
    actual, expected = [x.sort_values(keys).reset_index(drop=True) for x in (actual, expected)]
    for key in expected:
        same(actual[key].to_numpy(), expected[key].to_numpy(), path.name + '/' + key)


METRIC_FIELDS = ('n', 'ba', 'base_ba', 'accuracy', 'base_accuracy', 'repairs', 'harms', 'selected_n', 'selected_fraction')


def readout_tables(y, base, expert, action, identity):
    result = policy_readout(y, base, expert, action)
    row = {**identity, **{k: result['metrics'][k] for k in METRIC_FIELDS},
           'ba_gain': result['metrics']['ba_gain'], 'positive_eligible': action['positive_eligible'],
           'cap': action['cap'], 'cap_truncates': action['positive_eligible'] > action['cap']}
    cells = [{**identity, **{k: cell[k] for k in ('class_name', 'support', 'recall', 'base_recall', 'repairs', 'harms')}}
             for cell in result['class_metrics']]
    groups = [{**identity, 'group': 'agree' if cell['group'] == 'agreement' else 'disagree',
               **{k: cell[k] for k in ('n', 'repairs', 'harms', 'selected_n')}} for cell in result['group_metrics']]
    return row, cells, groups


def replay_fold(run, protocol, frame, number):
    parent = Path(protocol['input_run'])
    cfg = read_json(parent / 'config.json')
    origin, folder = parent / f'fold_{number:02d}', run / f'fold_{number:02d}'
    done, meta = read_json(origin / 'COMPLETE.json'), read_json(origin / 'roles.json')
    require(done['status'] == 'fold_artifacts_verified' and done['fingerprint'] == meta['fingerprint'] == cfg['fingerprint'],
            'parent fold not complete under the fixed configuration')
    roles, inputs = read_npz(origin / 'roles.npz'), read_npz(folder / 'inputs.npz')
    source, target = roles['correct_select'], roles['target']
    same(inputs['source_rows'], source, 'source rows')
    same(inputs['target_rows'], target, 'target rows')
    require(not np.intersect1d(source, target).size, 'source/target overlap')
    require(frame.iloc[target].port.eq(meta['port']).all(), 'target port differs')
    same(inputs['source_y'], frame.iloc[source].class_id.to_numpy(int), 'source labels')
    same(inputs['source_ports'], frame.iloc[source].port.to_numpy(str), 'source ports')
    priority = tie_priority(protocol['seed'], number, len(target))
    same(inputs['priority'], priority, 'shared tie priority')
    scores = read_npz(origin / 'target_scores.npz')
    same(scores['indexes'], target, 'parent target score rows')
    for arm in ('teacher',) + ARMS:
        completion, exited = read_json(origin / arm / 'COMPLETE.json'), read_json(origin / arm / 'exit.json')
        require(completion['training_completed'] is True and exited['exit_code'] == 0
                and completion['fingerprint'] == exited['fingerprint'] == cfg['fingerprint'], 'incomplete parent arm/' + arm)
        archive = read_npz(origin / arm / 'predictions.npz')
        expected = np.concatenate((roles['correct_fit'], source, target)) if arm == 'teacher' else np.concatenate((source, target))
        same(np.sort(archive['indexes']), np.sort(expected), 'parent archive identities/' + arm)
        for role, rows in [('S', source), ('T', target)]:
            require(np.array_equal(inputs[role + '_' + arm], align_logits(archive, rows)), 'inherited logits differ/' + role + '/' + arm)
        require(np.array_equal(scores[arm], inputs['T_' + arm]), 'parent target logits differ/' + arm)
    truth = read_npz(folder / 'readout.npz')
    y, base = frame.iloc[target].class_id.to_numpy(int), np.argmax(inputs['T_teacher'], axis=1)
    same(truth['labels'], y, 'target labels')
    same(truth['base'], base, 'target base')
    identity = {'seed': protocol['seed'], 'fold': number, 'port': meta['port']}
    tables = {k: [] for k in ('metrics', 'class_metrics', 'group_metrics', 'full_metrics', 'source_support')}
    base_source = inputs['S_teacher'].argmax(1)
    for arm in ('teacher',) + ARMS:
        expert = inputs['T_' + arm].argmax(1)
        full = policy_readout(y, base, expert, {'selected': expert != base, 'prediction': expert})
        tables['full_metrics'].append({**identity, 'arm': arm, **{k: full['metrics'][k] for k in METRIC_FIELDS}})
        if arm == 'teacher':
            actions = [('base', b, budget_policy(base, base, np.zeros(len(base)), b, priority)) for b in BUDGETS]
        else:
            fitted = pair_statistics(inputs['source_y'], base_source, inputs['S_' + arm].argmax(1), inputs['source_ports'])
            saved = read_npz(folder / f'pair_{arm}.npz')
            fields = dict(table='score', mass='denominator', total='numerator', weights='weights')
            require(set(saved) == set(fields), 'pair fields differ')
            for field, name in fields.items():
                same(saved[field], fitted[name], 'pair/' + arm + '/' + field)
            tables['source_support'].append({**identity, 'arm': arm, 'source_n': len(source),
                'source_classes': len(np.unique(inputs['source_y'])), 'observed_pairs': int((fitted['support'] > 0).sum()),
                'unseen_target_pair_n': int(np.sum(fitted['support'][base, expert] == 0))})
            score_map = gate_scores(inputs['T_teacher'], inputs['T_' + arm], fitted['score'])
            counts = np.bincount(y, minlength=8)
            score_map['ORACLE_ONLY'] = ((expert == y) & (base != y)) / (np.count_nonzero(counts) * counts[y])
            actions = []
            for gate, score in score_map.items():
                saved = read_npz(folder / f'policy_{arm}_{gate}.npz')
                require(set(saved) == {'score', 'candidate', *('selected_' + f'{round(100*b):02d}' for b in BUDGETS),
                                      *('prediction_' + f'{round(100*b):02d}' for b in BUDGETS)}, 'policy fields differ')
                same(saved['score'], score, arm + '/' + gate + '/score')
                same(saved['candidate'], expert, arm + '/' + gate + '/candidate')
                for budget in BUDGETS:
                    action = budget_policy(base, expert, score, budget, priority)
                    key = f'{round(100 * budget):02d}'
                    for field in ('selected', 'prediction'):
                        same(saved[field + '_' + key], action[field], arm + '/' + gate + '/' + field + '/' + key)
                    actions.append((gate, budget, action))
        for gate, budget, action in actions:
            row, cells, groups = readout_tables(y, base, expert, action, {**identity, 'arm': arm, 'gate': gate, 'budget': budget})
            tables['metrics'].append(row)
            tables['class_metrics'].extend(cells)
            tables['group_metrics'].extend(groups)
    require(len(tables['metrics']) == 123, 'policy count differs')
    print(f'Replayed pixel fold {number:02d}: 10 experts / 123 policies', flush=True)
    return tables


def aggregate(tables):
    performance = tables['metrics'].groupby(['arm', 'gate', 'budget']).agg(
        ports=('port', 'nunique'), ba=('ba', 'mean'), ba_gain=('ba_gain', 'mean'), accuracy=('accuracy', 'mean'),
        selected_fraction=('selected_fraction', 'mean'), truncated_ports=('cap_truncates', 'sum'),
        repairs=('repairs', 'sum'), harms=('harms', 'sum')).reset_index()
    pairs = [('repair_quality', 'repair'), ('repair_quality', 'repair_shuffle_global'),
             ('repair_quality', 'repair_shuffle_within'), ('repair_quality', 'ce_quality'),
             ('repair', 'ce'), ('repair', 'hard'), ('repair', 'repair_raw'), ('repair', 'dual_ab'),
             ('repair_quality', 'dual_ab')]
    comparisons = []
    for (fold, port, gate, budget), group in tables['metrics'].groupby(['fold', 'port', 'gate', 'budget']):
        if gate == 'base':
            continue
        values = group.set_index('arm')
        for left, right in pairs:
            comparisons.append(dict(fold=fold, port=port, gate=gate, budget=budget, left=left, right=right,
                ba_delta_pp=100 * (values.loc[left, 'ba'] - values.loc[right, 'ba']),
                accuracy_delta_pp=100 * (values.loc[left, 'accuracy'] - values.loc[right, 'accuracy']),
                oracle_only=gate == 'ORACLE_ONLY'))
    comparisons = pd.DataFrame(comparisons)
    summaries = comparisons.groupby(['gate', 'budget', 'left', 'right']).agg(
        ports=('port', 'nunique'), ba_delta_pp=('ba_delta_pp', 'mean'), accuracy_delta_pp=('accuracy_delta_pp', 'mean'),
        positive_ports=('ba_delta_pp', lambda a: int(np.sum(a > 1e-10))),
        negative_ports=('ba_delta_pp', lambda a: int(np.sum(a < -1e-10)))).reset_index()
    return {'performance': performance, 'contrasts': comparisons, 'contrast_summary': summaries,
            'primary': comparisons.query("gate == 'pair' and budget == .05 and left == 'repair_quality' and right == 'repair'").copy(),
            'full_performance': tables['full_metrics'].groupby('arm')[['ba', 'accuracy', 'base_ba', 'base_accuracy']].mean().reset_index()}


LIMITS = ['Development bridge, not a novel gate or blind confirmation.',
          'Partial-port aggregates are not full-24-port findings; no incomplete port is imputed.',
          'Source correct_select labels fit the pair gate after selecting expert epochs; not independent audit.',
          'Original pixel-pilot data, expert training and metadata are the trust boundary; no training replay or role reconstruction.',
          'All experts are evaluated before handoff; the budget controls changed predictions, not computation.',
          'ORACLE_ONLY uses target labels and is nondeployable; a larger ceiling is not evidence of learnable routing.',
          'Parent five-epoch training convergence and fair tuning remain separate questions.',
          'Quality controls, dual_ab and teacher retain the parent information/architecture/label-budget limitations.',
          'No new target fitting labels; historical development exposure remains.']


def verify_run(run, allow_smoke=False):
    run = Path(run)
    protocol = read_json(run / 'protocol.json')
    fixed = dict(experiment='bdh_pixel_handoff_v1', status='complete', expected_folds=24, arms=list(ARMS),
                 gates=list(GATES), budgets=list(BUDGETS), classes=list(CLASSES), target_fit_labels=0,
                 blind_confirmation=False, source_role='correct_select_reused_after_epoch_selection', tie_seed_tag=61801,
                 primary=PRIMARY, design='BDH_PIXEL_HANDOFF_PROTOCOL_20261003.md')
    for key, value in fixed.items():
        require(protocol[key] == value, 'protocol/' + key)
    folds = protocol['folds']
    require(folds and len(set(folds)) == len(folds) and set(folds) <= set(range(24)), 'invalid fold snapshot')
    require(protocol['completed_folds'] == len(folds) and protocol['partial'] is (len(folds) != 24), 'incomplete or mislabeled snapshot')
    require(protocol['smoke_only'] is bool(allow_smoke), 'smoke mode differs')
    parent = Path(protocol['input_run'])
    cfg = read_json(parent / 'config.json')
    require(not cfg['smoke'] and cfg['port_count'] == 24 and protocol['seed'] == cfg['seed'], 'parent pilot identity differs')
    require(Path(protocol['manifest']).resolve() == (parent.parent / 'inputs' / 'manifest.csv').resolve(), 'manifest provenance differs')
    frame = pd.read_csv(protocol['manifest'], dtype=str, keep_default_na=False)
    tables = {key: [] for key in ('metrics', 'class_metrics', 'group_metrics', 'full_metrics', 'source_support')}
    for number in folds:
        for key, rows in replay_fold(run, protocol, frame, number).items():
            tables[key].extend(rows)
    tables = {key: pd.DataFrame(rows) for key, rows in tables.items()}
    for key, table in tables.items():
        keys = ['seed', 'fold', 'port', 'arm'] if key in ('full_metrics', 'source_support') else KEY + (
            ['class_name'] if key == 'class_metrics' else ['group'] if key == 'group_metrics' else [])
        compare_csv(run / (key + '.csv'), table, keys)
    return protocol, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(status='running', input_run=str(args.run.resolve()), limits=LIMITS)
    try:
        protocol, tables = verify_run(args.run, args.allow_smoke)
        results = {**tables, **aggregate(tables)}
        for key, table in results.items():
            table.to_csv(args.output / (key + '.csv'), index=False)
        summary.update(status='complete', experiment='bdh_pixel_handoff_replay_v1', folds=protocol['folds'],
            expected_folds=24, partial=protocol['partial'], smoke_only=protocol['smoke_only'],
            experts_replayed=10 * len(protocol['folds']), policies_replayed=123 * len(protocol['folds']),
            primary=PRIMARY, blind_confirmation=False, target_fit_labels=0)
        print(f"Verified {summary['experts_replayed']} experts / {summary['policies_replayed']} policies; partial={summary['partial']}", flush=True)
    except Exception as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
