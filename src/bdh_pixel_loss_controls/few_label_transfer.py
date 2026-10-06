"""Frozen, historical-source simulation of few-label conditional-utility transfer."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bdh_pixel_loss_controls import training, utility_transport as parent_study
from bdh_pixel_loss_controls.few_label_split import annotation_order, split_port
from report_bdh_pixel_handoff import budget_policy, policy_readout, read_npz, require


OUTPUT = training.ROOT / 'artifacts/bdh_few_label_transfer_20261003'
LABEL_BUDGETS = (16, 32, 64)
DRAWS = 20
TAU = 8.
SEED = 20261003
GATES = ('source_destination', 'target_empirical', 'uniform_shrink_observed',
         'source_shrink_observed', 'uniform_shrink_robust', 'source_shrink_robust',
         'source_unadapted_robust')
KEYS = ['fold', 'arm', 'port', 'draw', 'label_budget', 'gate']


def conditional_estimates(labels, base, expert, source, tau=TAU):
    labels, base, expert = map(np.asarray, (labels, base, expert))
    require(labels.ndim == 1 and len(labels) and labels.shape == base.shape == expert.shape,
            'unaligned annotation sample')
    require(all(x.dtype.kind in 'iu' and np.all((x >= 0) & (x < 8))
                for x in (labels, base, expert)), 'invalid annotation class')
    require(source.shape == (8, 8, 8) and np.isfinite(source).all() and np.all(source >= 0)
            and np.allclose(source.sum(axis=(1, 2)), 1) and tau > 0, 'invalid source conditional')
    counts = np.bincount(64*labels + 8*base + expert, minlength=512).reshape(8, 8, 8)
    totals = counts.sum(axis=(1, 2))[:, None, None]
    empirical = np.divide(counts, totals, out=np.zeros((8, 8, 8)), where=totals > 0)
    return dict(counts=counts, observed=totals[:, 0, 0] > 0,
                empirical=empirical, uniform=(counts + tau/64)/(totals + tau),
                source=(counts + tau*source)/(totals + tau))


def utility_mass(conditional, observed, robust):
    conditional, observed = np.asarray(conditional), np.asarray(observed)
    require(conditional.shape == (8, 8, 8) and np.isfinite(conditional).all()
            and np.all(conditional >= 0), 'invalid conditional table')
    require(observed.shape == (8,) and observed.dtype == np.bool_ and observed.any(),
            'invalid observed support')
    support = observed[None, :]
    if robust:
        all_supports = ((np.arange(1, 256)[:, None] >> np.arange(8)) & 1).astype(bool)
        support = all_supports[np.all(all_supports[:, observed], axis=1)]
    b, e = np.indices((8, 8))
    values = (support[:, e]*conditional[e, b, e] - support[:, b]*conditional[b, b, e])
    values = values / support.sum(axis=1)[:, None, None]
    return values.min(axis=0)


def tables(stats, labels, base, expert, query_base, query_expert):
    fitted = conditional_estimates(labels, base, expert, stats['conditional'])
    pair = 8*query_base + query_expert
    require(len(pair) > 0, 'empty query')
    mass = np.bincount(pair, minlength=64).reshape(8, 8) / len(pair)
    numerators = dict(source_destination=stats['balanced_net'],
                      target_empirical=utility_mass(fitted['empirical'], fitted['observed'], False),
                      source_unadapted_robust=utility_mass(stats['conditional'], fitted['observed'], True))
    for prior in ('uniform', 'source'):
        for mode in ('observed', 'robust'):
            numerators[f'{prior}_shrink_{mode}'] = utility_mass(
                fitted[prior], fitted['observed'], mode == 'robust')
    scores = np.stack([np.divide(numerators[g], mass, out=np.zeros((8, 8)), where=mass > 0)
                       for g in GATES])
    return scores, fitted


def prepare(output, parents):
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(experiment='bdh_few_label_transfer_v1', parents=[str(p['path']) for p in parents],
                label_budgets=list(LABEL_BUDGETS), draws=DRAWS, tau=TAU, gates=list(GATES),
                handoff_budget=.05, primary_label_budget=64, primary_mode='out_of_port',
                primary_contrast='source_shrink_robust minus uniform_shrink_robust',
                seed=SEED, split_seed_offset=83001, draw_seed_offset=84001, tie_seed_offset=85001,
                annotation_unit='one chip label; no vessel propagation',
                historical_development=True, independent_confirmation=False,
                target_labels_used=0, target_predictions=0, gpu_training_steps=0)
    training.write_json(output / 'plan.json', plan)
    splits, summary = [], []
    for fold, parent in enumerate(parents):
        rows, g = parent['rows'], parent['roles']['gate_fit']
        ports = np.array([rows[i]['port'] for i in g])
        for ordinal, port in enumerate(np.unique(ports)):
            split = split_port(rows, g[ports == port], SEED + 83001 + 100*fold + ordinal)
            require(len(split['adaptation_pool']) >= max(LABEL_BUDGETS) and len(split['query']),
                    f'{port}: frozen split cannot support 64 labels and nonempty query; no redraw')
            filename = f'split_{fold}_{ordinal}.npz'
            np.savez_compressed(output / filename, **split)
            item = dict(fold=fold, port=str(port), ordinal=ordinal, file=filename)
            splits.append(item)
            summary.append({**item, **{k + '_n': len(v) for k, v in split.items()}})
    training.write_json(output / 'splits_saved.json', splits)
    pd.DataFrame(summary).to_csv(output / 'split_summary.csv', index=False)
    return splits


def positions(parent, indexes):
    g = parent['roles']['gate_fit']
    order = np.argsort(g)
    pos = np.searchsorted(g[order], indexes)
    require(np.all(pos < len(g)) and np.array_equal(g[order[pos]], indexes), 'indexes outside G')
    return order[pos]


def freeze(output, parents, splits):
    stats = {}
    for fold, parent in enumerate(parents):
        b = parent['roles']['correct_fit']
        y = parent_study.handoff.decode_labels(parent['rows'], b)
        ports = np.array([parent['rows'][i]['port'] for i in b])
        for arm, expert in parent['experts'].items():
            stats[fold, arm] = parent_study.source_statistics(y, parent['base']['fit'], expert['out_of_port'], ports)
            np.savez_compressed(output / f'source_{fold}_{arm}.npz', **stats[fold, arm])
    inventory = []
    for item in splits:
        fold, ordinal = item['fold'], item['ordinal']
        parent = parents[fold]
        split = read_npz(output / item['file'])
        query_pos = positions(parent, split['query'])
        qb = parent['base']['gate'][query_pos]
        priority = np.argsort(np.random.default_rng(SEED + 85001 + 100*fold + ordinal).permutation(len(qb)))
        for draw in range(DRAWS):
            sampled = annotation_order(split['adaptation_pool'], SEED + 84001 + 1000*fold + 100*ordinal + draw)
            for budget in LABEL_BUDGETS:
                annotation = sampled[:budget]
                labels = parent_study.handoff.decode_labels(parent['rows'], annotation)
                annotation_pos = positions(parent, annotation)
                ab = parent['base']['gate'][annotation_pos]
                for arm, expert in parent['experts'].items():
                    ae, qe = expert['gate'][annotation_pos], expert['gate'][query_pos]
                    score_tables, fitted = tables(stats[fold, arm], labels, ab, ae, qb, qe)
                    score = score_tables[:, qb, qe]
                    masks = np.stack([budget_policy(qb, qe, s, .05, priority)['selected'] for s in score])
                    filename = f'policy_{fold}_{ordinal}_{draw:02d}_{budget}_{arm}.npz'
                    np.savez_compressed(output / filename, indexes=split['query'], annotation_indexes=annotation,
                                        annotation_labels=labels, annotation_base=ab, annotation_expert=ae,
                                        base=qb, expert=qe, priority=priority, tables=score_tables,
                                        score=score, selected=masks, gates=np.array(GATES),
                                        counts=fitted['counts'], observed=fitted['observed'])
                    inventory.append(dict(fold=fold, port=item['port'], ordinal=ordinal, draw=draw,
                                          label_budget=budget, arm=arm, file=filename))
    training.write_json(output / 'policies_saved.json', inventory)
    return inventory


def readout(output, parents, inventory):
    require(len(inventory)*len(GATES) == 3360 and (output / 'policies_saved.json').exists(),
            'all policies must be saved before query labels are decoded')
    records, classes, support_records = [], [], []
    seen = set()
    for item in inventory:
        saved = read_npz(output / item['file'])
        parent = parents[item['fold']]
        labels = parent_study.handoff.decode_labels(parent['rows'], saved['indexes'])
        support = np.bincount(labels, minlength=8) > 0
        observed = saved['observed']
        missing = support & ~observed
        identity = {k: item[k] for k in KEYS if k != 'gate'}
        support_key = item['fold'], item['port'], item['draw'], item['label_budget']
        if support_key not in seen:
            seen.add(support_key)
            annotations = saved['annotation_indexes']
            support_records.append({k: item[k] for k in ('fold', 'port', 'draw', 'label_budget')} |
                dict(observed_classes=int(observed.sum()), query_classes=int(support.sum()),
                     missing_query_classes=int(missing.sum()),
                     missing_query_fraction=float(np.mean(missing[labels])),
                     observed_absent_from_query=int(np.sum(observed & ~support)),
                     query_support_in_uncertainty_set=bool(np.all(support[observed])),
                     annotation_vessels=len({parent['rows'][i]['mmsi'] for i in annotations}),
                     annotation_products=len({parent['rows'][i]['product'] for i in annotations})))
        for gate_index, gate in enumerate(GATES):
            policy = budget_policy(saved['base'], saved['expert'], saved['score'][gate_index], .05, saved['priority'])
            require(np.array_equal(policy['selected'], saved['selected'][gate_index]), 'saved policy changed')
            result = policy_readout(labels, saved['base'], saved['expert'], policy)
            key = {**identity, 'gate': gate}
            records.append({**key, 'annotation_labels_used': 0 if gate == 'source_destination' else item['label_budget'],
                            'handoff_budget': .05, **result['metrics'], 'cap': policy['cap'],
                            'positive_eligible': policy['positive_eligible']})
            classes.extend({**key, **cell} for cell in result['class_metrics'])
    return pd.DataFrame(records), pd.DataFrame(classes), pd.DataFrame(support_records)


def summarize(output, metrics, classes, supports):
    group_keys = ['arm', 'port', 'label_budget', 'gate']
    fields = ['ba_gain', 'selected_n', 'selected_fraction', 'repairs', 'harms']
    port_means = metrics.groupby(group_keys, as_index=False)[fields].mean()
    means = port_means.groupby(['arm', 'label_budget', 'gate'], as_index=False)[fields].mean()
    paired = metrics.pivot(index=['fold', 'arm', 'port', 'draw', 'label_budget'], columns='gate', values='ba_gain')
    contrasts = paired.reset_index()[['fold', 'arm', 'port', 'draw', 'label_budget']].copy()
    comparisons = [('source_shrink_robust', 'uniform_shrink_robust'),
                   ('source_shrink_robust', 'source_destination'),
                   ('source_shrink_robust', 'target_empirical'),
                   ('source_shrink_robust', 'source_unadapted_robust'),
                   ('source_shrink_robust', 'source_shrink_observed'),
                   ('source_shrink_observed', 'uniform_shrink_observed')]
    for left, right in comparisons:
        contrasts[left + '_minus_' + right + '_pp'] = 100*(paired[left]-paired[right]).to_numpy()
    primary_column = 'source_shrink_robust_minus_uniform_shrink_robust_pp'
    primary_ports = contrasts[contrasts.label_budget == 64].groupby(['arm', 'port'])[primary_column].mean()
    primary = primary_ports.groupby('arm').mean()
    for name, frame in [('metrics', metrics), ('class_metrics', classes), ('support_diagnostics', supports),
                        ('port_means', port_means), ('means', means), ('contrasts', contrasts)]:
        frame.to_csv(output / f'{name}.csv', index=False)
    require(len(metrics) == 3360 and len(classes) == 26880 and len(supports) == 240, 'incomplete fixed grid')
    complete = dict(status='complete', policy_readouts=len(metrics), class_metric_rows=len(classes),
                    primary_contrasts_pp=primary.to_dict(), primary_both_arms_positive=bool((primary > 0).all()),
                    label_budgets_per_pipeline=list(LABEL_BUDGETS), annotation_unit='chip',
                    simulated_annotation=True, historical_development=True, independent_confirmation=False,
                    target_labels_used=0, target_predictions=0, gpu_training_steps=0, novelty_established=False)
    training.write_json(output / 'COMPLETE.json', complete)
    return complete


def run(output):
    output = Path(output).resolve()
    require(not output.exists(), 'use a new output; do not overwrite a completed diagnostic')
    parents = [parent_study.load_parent(path) for path in parent_study.PARENTS]
    splits = prepare(output, parents)
    inventory = freeze(output, parents, splits)
    metrics, classes, supports = readout(output, parents, inventory)
    return summarize(output, metrics, classes, supports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output)), flush=True)
