"""Fixed source-utility / destination-count decomposition, with no model training."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bdh_pixel_loss_controls import crossfit_handoff as handoff
from bdh_pixel_loss_controls import source_role_plan as allocation
from bdh_pixel_loss_controls import training
from report_bdh_pixel_handoff import (
    align_logits, budget_policy, policy_readout, read_npz, require, source_weights,
)


ROOT = training.ROOT
PARENTS = (ROOT / 'artifacts/bdh_crossfit_handoff_20261003',
           ROOT / 'artifacts/bdh_crossfit_replication_20261003/fold_01/handoff')
OUTPUT = ROOT / 'artifacts/bdh_utility_transport_20261003'
GATES = ('source_count', 'destination_count', 'class_uniform_source_count',
         'class_uniform_destination_count')


def predictions(base, expert):
    base, expert = np.asarray(base), np.asarray(expert)
    require(base.ndim == 1 and base.shape == expert.shape and len(base) > 0,
            'empty or unaligned predictions')
    require(all(v.dtype.kind in 'iu' and np.all((v >= 0) & (v < 8)) for v in (base, expert)),
            'invalid predictions')
    return base, expert


def source_statistics(labels, base, expert, ports):
    base, expert = predictions(base, expert)
    labels, ports = np.asarray(labels), np.asarray(ports)
    weights = source_weights(labels, ports)
    require(labels.shape == base.shape and np.array_equal(np.unique(labels), np.arange(8)),
            'source requires all eight classes')
    pair = 8 * base + expert
    joint = np.bincount(64 * labels + pair, weights=weights, minlength=512).reshape(8, 8, 8)
    class_mass = joint.sum(axis=(1, 2))
    conditional = joint / class_mass[:, None, None]
    mass = np.empty(len(labels), dtype=float)
    names = np.unique(ports)
    for name in names:
        use = ports == name
        mass[use] = 1. / (len(names) * use.sum())
    sample_mass = np.bincount(pair, weights=mass, minlength=64).reshape(8, 8)
    # Preserve the parent table's accumulation order, including numerical ties.
    delta = (expert == labels).astype(float) - (base == labels).astype(float)
    balanced_net = np.bincount(pair, weights=weights * delta, minlength=64).reshape(8, 8)
    b, e = np.indices((8, 8))
    uniform_net = (conditional[e, b, e] - conditional[b, b, e]) / 8
    return dict(joint=joint, class_mass=class_mass, conditional=conditional,
                sample_mass=sample_mass, balanced_net=balanced_net, uniform_net=uniform_net)


def divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros((8, 8)), where=denominator > 0)


def score_tables(stats, query_base, query_expert, support=None):
    b, e = predictions(query_base, query_expert)
    mass = np.bincount(8 * b + e, minlength=64).reshape(8, 8) / len(b)
    result = dict(source_count=divide(stats['balanced_net'], stats['sample_mass']),
                  destination_count=divide(stats['balanced_net'], mass),
                  class_uniform_source_count=divide(stats['uniform_net'], stats['sample_mass']),
                  class_uniform_destination_count=divide(stats['uniform_net'], mass))
    if support is not None:
        support = np.asarray(support)
        require(support.shape == (8,) and support.dtype == np.bool_ and support.any(), 'invalid G support')
        b, e = np.indices((8, 8))
        q = stats['conditional']
        numerator = (support[e] * q[e, b, e] - support[b] * q[b, b, e]) / support.sum()
        result['ORACLE_G_SUPPORT'] = divide(numerator, mass)
    return result


def load_parent(path):
    complete = training.read_json(path / 'COMPLETE.json')
    require(complete['status'] == 'complete' and complete['policy_readouts'] == 108,
            'requires completed source handoff')
    require(complete['target_labels_used'] == complete['target_predictions'] == 0, 'target parent use')
    study = Path(complete['plan']['study'])
    plan = training.read_json(study / 'plan.json')
    roles = read_npz(study / 'roles.npz')
    b, g = roles['correct_fit'], roles['gate_fit']
    require(not np.intersect1d(np.r_[b, g], roles['target']).size, 'target source overlap')
    rows = allocation.load_manifest(Path(plan['data']) / 'manifest.csv')
    archive = read_npz(Path(plan['teacher']) / 'source_teacher_predictions.npz')
    base = dict(fit=align_logits(archive, b).argmax(1), gate=align_logits(archive, g).argmax(1))
    experts = {}
    for arm in handoff.generation.ARMS:
        full = read_npz(study / 'full' / arm / 'source_predictions.npz')
        outside = read_npz(study / f'{arm}_out_of_port.npz')
        experts[arm] = dict(in_sample=align_logits(full, b).argmax(1),
                            out_of_port=align_logits(outside, b).argmax(1),
                            gate=align_logits(full, g).argmax(1))
    return dict(path=path, plan=complete['plan'], roles=roles, rows=rows, base=base, experts=experts)


def freeze_ordinary(output, parents):
    output.mkdir(parents=True, exist_ok=False)
    training.write_json(output / 'plan.json', dict(experiment='bdh_utility_transport_v1',
                        parents=[str(p['path']) for p in parents], gates=list(GATES),
                        budgets=list(handoff.BUDGETS), primary_budget=.05,
                        primary_mode='out_of_port', primary_contrast='destination_count minus source_count',
                        gpu_training_steps=0, target_labels_used=0, target_predictions=0,
                        source_readout_historically_blind=False, oracle_support_not_deployable=True))
    queries = []
    for fold, parent in enumerate(parents):
        rows, roles = parent['rows'], parent['roles']
        b, g = roles['correct_fit'], roles['gate_fit']
        y = handoff.decode_labels(rows, b)
        ports = np.array([rows[i]['port'] for i in b])
        gports = np.array([rows[i]['port'] for i in g])
        for arm, expert in parent['experts'].items():
            for mode in handoff.MODES:
                stats = source_statistics(y, parent['base']['fit'], expert[mode], ports)
                np.savez_compressed(output / f'source_{fold}_{arm}_{mode}.npz', **stats)
                for ordinal, port in enumerate(np.unique(gports)):
                    use = gports == port
                    qb, qe = parent['base']['gate'][use], expert['gate'][use]
                    priority = np.argsort(np.random.default_rng(np.random.SeedSequence(
                        [parent['plan']['seed'], ordinal, parent['plan']['tie_tag']])).permutation(use.sum()))
                    for gate, table in score_tables(stats, qb, qe).items():
                        arrays = dict(indexes=g[use], base=qb, expert=qe, priority=priority,
                                      score=table[qb, qe], table=table)
                        for budget in handoff.BUDGETS:
                            policy = budget_policy(qb, qe, arrays['score'], budget, priority)
                            arrays[f'selected_{round(100 * budget):02d}'] = policy['selected']
                        filename = f'policy_{fold}_{arm}_{mode}_{ordinal}_{gate}.npz'
                        np.savez_compressed(output / filename, **arrays)
                        queries.append(dict(fold=fold, arm=arm, mode=mode, port=str(port),
                                            gate=gate, ordinal=ordinal, file=filename))
    training.write_json(output / 'ordinary_policies_saved.json', queries)
    return queries


def readout(output, parents, queries):
    records, classes, conditional_rows = [], [], []
    oracle_done = set()
    old_metrics = [pd.read_csv(p['path'] / 'metrics.csv') for p in parents]
    baseline_checks = 0
    for query in queries:
        saved = read_npz(output / query['file'])
        parent = parents[query['fold']]
        y = handoff.decode_labels(parent['rows'], saved['indexes'])
        b, e = saved['base'], saved['expert']
        identity = {k: query[k] for k in ('fold', 'arm', 'mode', 'port', 'gate')}
        scores = {query['gate']: saved['score']}
        key = query['fold'], query['arm'], query['mode'], query['port']
        if key not in oracle_done:
            oracle_done.add(key)
            stats = read_npz(output / f"source_{query['fold']}_{query['arm']}_{query['mode']}.npz")
            support = np.bincount(y, minlength=8) > 0
            table = score_tables(stats, b, e, support)['ORACLE_G_SUPPORT']
            scores['ORACLE_G_SUPPORT'] = table[b, e]
            np.savez_compressed(output / ('oracle_support_' + query['file']), indexes=saved['indexes'],
                                support=support, table=table, score=table[b, e])
            for category in np.flatnonzero(support):
                use = y == category
                empirical = np.bincount(8*b[use] + e[use], minlength=64).reshape(8, 8) / use.sum()
                conditional_rows.append(dict(fold=query['fold'], arm=query['arm'], mode=query['mode'],
                                             port=query['port'], class_id=int(category), G_support=int(use.sum()),
                                             empirical_conditional_tv=float(.5*np.abs(empirical-stats['conditional'][category]).sum()),
                                             G_mass_on_source_zero_cells=float(empirical[stats['conditional'][category] == 0].sum())))
        for gate, score in scores.items():
            for budget in handoff.BUDGETS:
                policy = budget_policy(b, e, score, budget, saved['priority'])
                if gate != 'ORACLE_G_SUPPORT':
                    require(np.array_equal(policy['selected'], saved[f'selected_{round(100 * budget):02d}']),
                            'saved ordinary mask differs')
                result = policy_readout(y, b, e, policy)
                row = {**identity, 'gate': gate, 'budget': budget,
                       'uses_G_labels_for_score': gate == 'ORACLE_G_SUPPORT',
                       **result['metrics'], 'cap': policy['cap'], 'positive_eligible': policy['positive_eligible']}
                records.append(row)
                classes.extend({**identity, 'gate': gate, 'budget': budget, **cell} for cell in result['class_metrics'])
                if gate == 'source_count':
                    previous = old_metrics[query['fold']]
                    previous = previous[(previous.arm == query['arm']) & (previous.port == query['port']) &
                                        (previous.gate == query['mode'] + '_pair_count') &
                                        np.isclose(previous.budget, budget)].iloc[0]
                    for field in result['metrics']:
                        require(np.isclose(previous[field], row[field], rtol=1e-10, atol=1e-12),
                                'parent count readout differs: ' + field)
                    old = read_npz(parent['path'] / f"policy_{query['arm']}_{query['ordinal']:02d}_{query['mode']}_pair_count.npz")
                    require(np.array_equal(old[f'selected_{round(100 * budget):02d}'], policy['selected']),
                            'parent count mask differs')
                    baseline_checks += 1
    return pd.DataFrame(records), pd.DataFrame(classes), pd.DataFrame(conditional_rows), baseline_checks


def run(output):
    output = Path(output).resolve()
    parents = [load_parent(path) for path in PARENTS]
    queries = freeze_ordinary(output, parents)
    metrics, classes, conditionals, checks = readout(output, parents, queries)
    require(len(metrics) == 240 and checks == 48, 'incomplete diagnostic grid')
    contrasts = []
    for key, group in metrics.groupby(['fold', 'arm', 'mode', 'port', 'budget']):
        gains = group.set_index('gate')['ba_gain']
        for gate in list(GATES[1:]) + ['ORACLE_G_SUPPORT']:
            contrasts.append(dict(zip(['fold', 'arm', 'mode', 'port', 'budget'], key)) |
                             dict(gate=gate, gain_pp=100*gains[gate],
                                  minus_source_count_pp=100*(gains[gate]-gains['source_count'])))
    contrasts = pd.DataFrame(contrasts)
    means = contrasts.groupby(['arm', 'mode', 'budget', 'gate'], as_index=False)[
        ['gain_pp', 'minus_source_count_pp']].mean()
    primary = means[(means['mode'] == 'out_of_port') & np.isclose(means.budget, .05) &
                    (means.gate == 'destination_count')]
    for name, frame in [('metrics', metrics), ('class_metrics', classes), ('contrasts', contrasts),
                        ('means', means), ('conditional_diagnostics', conditionals)]:
        frame.to_csv(output / f'{name}.csv', index=False)
    complete = dict(status='complete', policy_readouts=len(metrics), deployable_readouts=192,
                    oracle_support_readouts=48, parent_count_readouts_reproduced=checks,
                    primary_both_arms_positive=bool((primary.minus_source_count_pp > 0).all()),
                    primary_contrasts_pp=dict(zip(primary.arm, primary.minus_source_count_pp)),
                    target_labels_used=0, target_predictions=0, gpu_training_steps=0,
                    independent_confirmation=False, novelty_established=False)
    training.write_json(output / 'COMPLETE.json', complete)
    return complete


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output)), flush=True)
