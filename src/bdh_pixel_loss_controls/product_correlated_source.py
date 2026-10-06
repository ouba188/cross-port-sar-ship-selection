"""Source-fitted acquisition correlation changes same64-label conditional updates."""

import os

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from bdh_pixel_loss_controls import conditional_likelihood as likelihood
from bdh_pixel_loss_controls import product_correlated_likelihood as model
from bdh_pixel_loss_controls import product_correlation_source_fit as fit_run
from bdh_pixel_loss_controls import conditional_likelihood_gate as gate_loader
from bdh_pixel_loss_controls import action_prototype_source_pilot as shared
from bdh_pixel_loss_controls import training, rank_profile_transport
from report_bdh_pixel_handoff import read_npz, require


PARTIAL = training.ROOT / 'artifacts/bdh_product_correlated_20261004'
OUTPUT = training.ROOT / 'artifacts/bdh_product_correlated_20261004_v2'
ARMS = ('flat', 'target_likelihood', 'product_likelihood')


def fixed_count(base, expert, score, ranking, priority, count):
    eligible = np.flatnonzero(base != expert)
    require(0 <= count <= len(eligible), 'fixed count exceeds disagreements')
    order = eligible[np.lexsort((priority[eligible], -ranking[eligible], -score[eligible]))]
    mask = np.zeros(len(base), dtype=bool)
    mask[order[:count]] = True
    return mask


def freeze_stage(stage, parent):
    output = OUTPUT / stage
    output.mkdir()
    previous = training.read_json(parent / 'POLICIES_SAVED.json')
    entries = [entry for entry in previous if entry['arm'] == 'target_likelihood']
    inventory, diagnostics = [], []
    role = 'expert_select' if stage == 'E' else 'gate_fit'
    for fold in (0, 1):
        with np.load(training.ROOT / f'artifacts/bdh_joint_modules_20261004_v2/fold_{fold}/{role}.npz', allow_pickle=False) as archive:
            meta = {key: archive[key] for key in ('indexes', 'product')}
        lookup = {int(value): i for i, value in enumerate(meta['indexes'])}
        for seed in (20261005, 20261006):
            directory = output / f'seed_{seed}/fold_{fold}'
            directory.mkdir(parents=True)
            old_directory = parent / f'seed_{seed}/fold_{fold}'
            current = read_npz(old_directory / f'{stage}_features.npz')
            require(np.array_equal(current['indexes'], meta['indexes']), 'stage feature identities differ')
            for ordinal in (0, 1):
                source = read_npz(old_directory / f'source_{ordinal}.npz')
                correlation = read_npz(fit_run.OUTPUT / stage / f'seed_{seed}/fold_{fold}/correlation_{ordinal}.npz')
                require(np.array_equal(correlation['source_indexes'], source['source_indexes']), 'source correlation rows differ')
                rho = float(correlation['rho'])
                selected = [entry for entry in entries if entry['seed'] == seed and entry['fold'] == fold and entry['ordinal'] == ordinal]
                require(len(selected) == 20, 'twenty draws required')
                for item in selected:
                    old = read_npz(parent / item['file'])
                    ap = np.array([lookup[int(i)] for i in old['annotation_indexes']])
                    qp = np.array([lookup[int(i)] for i in old['indexes']])
                    require(len(ap) == 64 and not np.intersect1d(meta['product'][ap], meta['product'][qp]).size,
                            'new-product predictive model requires disjoint annotation/query products')
                    partial_path = PARTIAL / stage / f'seed_{seed}/fold_{fold}/product_{ordinal}_{item["draw"]:02d}.npz'
                    reused = partial_path.exists()
                    if reused:
                        previous_fit = read_npz(partial_path)
                        for key, expected in dict(annotation_indexes=old['annotation_indexes'], annotation_labels=old['annotation_labels'],
                                                  annotation_products=meta['product'][ap], query_indexes=old['indexes'], rho=np.array(rho)).items():
                            require(np.array_equal(previous_fit[key], expected), 'partial fit input changed: ' + key)
                        predicted = {key: previous_fit[key] for key in ('means', 'inflation', 'loglikelihood', 'class_covariance', 'counts')}
                        fitted = {key: previous_fit[key] for key in ('prior', 'posterior', 'ratio', 'objective', 'fixed_point_error')}
                    else:
                        predicted = model.predict(source, current['features'][ap], old['annotation_labels'],
                                                  meta['product'][ap], current['features'][qp], rho)
                        fitted = likelihood.class_mass(predicted['loglikelihood'], predicted['counts'])
                    score = likelihood.robust_scores(fitted['ratio'], old['base'], old['expert'], old['observed'])
                    natural = rank_profile_transport.policy(old['base'], old['expert'], score, old['ranking'], old['priority'])
                    k = int(old['common_k'])
                    matched = fixed_count(old['base'], old['expert'], score, old['ranking'], old['priority'], k)
                    np.savez_compressed(directory / f'product_{ordinal}_{item["draw"]:02d}.npz',
                        annotation_indexes=old['annotation_indexes'], annotation_labels=old['annotation_labels'],
                        annotation_products=meta['product'][ap], query_indexes=old['indexes'], rho=np.array(rho), **predicted, **fitted)
                    diagnostics.append(dict(stage=stage, seed=seed, fold=fold, ordinal=ordinal, draw=item['draw'], port=item['port'],
                        rho=rho, annotation_products=len(np.unique(meta['product'][ap])),
                        maximum_class_cross_covariance=float(np.max(np.abs(predicted['class_covariance'] - np.diag(np.diag(predicted['class_covariance']))))),
                        fixed_point_error=float(fitted['fixed_point_error']), natural_count=int(natural['selected'].sum()),
                        previous_natural_count=int(old['selected'][0].sum()), common_k=k, reused_fit=reused,
                        matched_nonpositive_selected=int(np.count_nonzero(matched & (score <= 1e-12))),
                        matched_exchanged_rows=int(np.count_nonzero(matched != old['selected'][1]))))
                    for arm in ARMS:
                        filename = f'policy_{seed}_{fold}_{ordinal}_{item["draw"]:02d}_{arm}.npz'
                        if arm == 'product_likelihood':
                            saved = dict(old, utility=score, selected=np.stack((natural['selected'], matched)))
                        elif arm == 'target_likelihood':
                            saved = old
                        else:
                            saved = read_npz(parent / filename)
                        require(int(saved['common_k']) == k and int(saved['selected'][1].sum()) == k, 'inherited K differs')
                        np.savez_compressed(output / filename, **saved)
                        inventory.append(dict(item, arm=arm, file=filename))
                print(f'product {stage} seed={seed} fold={fold} context={ordinal} frozen', flush=True)
    require(len(inventory) == 480, 'incomplete stage policies')
    training.write_json(output / 'POLICIES_SAVED.json', inventory)
    training.write_json(output / 'FIT_DIAGNOSTICS.json', diagnostics)
    return inventory


def main():
    require(training.read_json(fit_run.OUTPUT / 'COMPLETE.json')['source_fits'] == 16, 'source correlation fit incomplete')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    training.write_json(OUTPUT / 'plan.json', dict(experiment='acquisition_correlated_evidence_update',
        parents={k: str(v) for k, v in fit_run.PARENTS.items()}, correlation_parent=str(fit_run.OUTPUT),
        stages=['E', 'G'], arms=list(ARMS), seeds=[20261005, 20261006], folds=[0, 1], annotation_budget=64,
        tau=8., query_weight=1., em_steps=128, replacement_fraction=.05,
        source_correlation='source-only restricted likelihood with port/class fixed means; no BA/rho grid',
        annotation_model='correlated Gaussian with shared product disturbance; joint class posterior, not count scaling',
        predictive_model='new query products independent of annotation products, same total source covariance',
        partial_parent=str(PARTIAL), partial_stop='positive-only inheritedK infeasible; no new query readout',
        matched_count='same inheritedK, rank all disagreements including nonpositive scores; never lower baselineK',
        natural_count='unchanged positive-only 5percent cap',
        reused_fits='completed partial fit arrays retained after input/identity equality; only missing fits computed',
        no_power_cap=True, fixed_before_new_readout=True, source_gaussian_refits=0, new_neural_training=0,
        primary='product-flat matchedBA strictly positive in both seeds separately E/G',
        secondary_required='equal-stage product-previous-target matchedBA positive in each seed; natural product-flat positive both stages/seeds',
        historical_E_and_G_exposure=True, T_readouts=0, independent_confirmation=False, novelty_established=False))
    inventories = {stage: freeze_stage(stage, parent) for stage, parent in fit_run.PARENTS.items()}
    training.write_json(OUTPUT / 'POLICIES_SAVED.json', inventories)
    parents = [gate_loader.load_fold(fold) for fold in (0, 1)]
    contrasts = []
    for stage, inventory in inventories.items():
        output = OUTPUT / stage
        result = shared.readout(parents, inventory, output, 'product_likelihood', 'target_likelihood')
        table = pd.read_csv(output / 'seed_contrasts.csv')
        result['source_promotion_criterion_passed'] = bool((table[table.budget_mode == 'matched'].product_likelihood_minus_flat_pp > 0).all())
        training.write_json(output / 'COMPLETE.json', dict(status='complete', stage=stage, **result, correlated_updates=160))
        contrasts.append(table.assign(stage=stage))
    table = pd.concat(contrasts, ignore_index=True)
    table.to_csv(OUTPUT / 'stage_seed_contrasts.csv', index=False)
    columns = ['product_likelihood_minus_target_likelihood_pp', 'product_likelihood_minus_flat_pp']
    average = table.groupby(['seed', 'budget_mode'], as_index=False)[columns].mean()
    average.to_csv(OUTPUT / 'combined_seed_contrasts.csv', index=False)
    matched, averaged = table[table.budget_mode == 'matched'], average[average.budget_mode == 'matched']
    natural = table[table.budget_mode == 'natural']
    passed = bool((matched.product_likelihood_minus_flat_pp > 0).all()
        and (averaged.product_likelihood_minus_target_likelihood_pp > 0).all()
        and (natural.product_likelihood_minus_flat_pp > 0).all())
    diagnostics = [record for stage in fit_run.PARENTS for record in training.read_json(OUTPUT / stage / 'FIT_DIAGNOSTICS.json')]
    training.write_json(OUTPUT / 'COMPLETE.json', dict(status='complete', native_pid=os.getpid(),
        correlated_updates=320, policy_files=960, readouts=1920, class_readouts=15360,
        source_promotion_criterion_passed=passed,
        positive_vs_flat_all_stage_seeds=bool((matched.product_likelihood_minus_flat_pp > 0).all()),
        positive_vs_previous_equal_stage_each_seed=bool((averaged.product_likelihood_minus_target_likelihood_pp > 0).all()),
        positive_vs_flat_natural_all_stage_seeds=bool((natural.product_likelihood_minus_flat_pp > 0).all()),
        reused_correlated_updates=sum(int(record['reused_fit']) for record in diagnostics),
        new_correlated_updates=sum(int(not record['reused_fit']) for record in diagnostics),
        matched_nonpositive_selected=sum(record['matched_nonpositive_selected'] for record in diagnostics),
        source_gaussian_refits=0, new_neural_training=0, new_neural_predictions=0, T_readouts=0,
        historical_E_and_G_exposure=True, independent_confirmation=False, novelty_established=False))
    print(average.to_string(index=False), flush=True)
    print(f'COMPLETE {OUTPUT}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
