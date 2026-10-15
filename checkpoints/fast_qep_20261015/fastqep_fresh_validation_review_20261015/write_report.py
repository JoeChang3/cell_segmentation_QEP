"""Render a report from the audited fresh-sample result, without new inference."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
a = json.loads((HERE/'analysis.json').read_text())
decision = a['decision']
lines = ['FAST-QEP FROZEN-WEIGHT FRESH-SAMPLE VALIDATION', '2026-10-15', '',
         'EXECUTED SCOPE',
         'New MCMC seeds, same two development-image first tiles and same GP-fitted',
         'parameter points. q=1.5; four chains per target; 2000 warmup plus 4000',
         'retained draws per chain. Scores evaluated every tenth retained draw.',
         'Control weights were fitted exclusively on OLD pilot draws and frozen.',
         'No weight retuning, parameter optimization, new q, or segmentation evaluation.',
         'Independent Monte Carlo replication does not mean independent image validation.', '',
         'Run status: '+a['run_status'],
         f"Runtime: {a['runtime_seconds']:.2f} seconds ({a['runtime_seconds']/60:.2f} minutes).",
         'All numerical/convergence gates pass: '+str(a['all_numerical_gates_pass']),
         'Precision repair supported in both datasets: '+str(a['precision_repair_supported_both_datasets']),
         'Predeclared decision: '+decision, '',
         'FROZEN DECISION RULE',
         'Retain original sampler, latent, score and mean-MCSE gates; additionally',
         'weighted-score Rhat<1.01 and bulk/tail ESS>400 for every coordinate.',
         'Precision: weighted MCSE <=1.2 times raw MCSE for all coordinates, and',
         '<=0.8 times original fixed-control MCSE for both kernel coordinates.',
         'Direction: |QEP score-GP score|>3 MCSE, scaled absolute difference>1e-8,',
         'and all four fresh chain differences have the same sign.',
         'This is a descriptive numerical screen, not a p-value, multiplicity-adjusted',
         'test, proof of equality or equivalence test. No post-hoc changes to rules.', '']
for t in a['targets']:
    lines += ['TARGET: '+t['dataset'],
              'Numerical/convergence pass: '+str(t['numerical_convergence_pass']),
              'Precision repair pass: '+str(t['precision_repair_supported']),
              'Fresh seeds: '+str(t['fresh_seed_components']),
              'Weights (m, log s, log nu, log beta row, log beta column): '+str(t['weights']),
              'Candidate directions: '+str(t['candidate_directions']),
              f"Maximum |QEP-GP|/MCSE: {t['maximum_abs_difference_over_mcse']:.6f}",
              f"Maximum weighted Rhat: {t['maximum_weighted_rhat']:.6f}; minimum bulk/tail ESS: "
              f"{t['minimum_weighted_bulk_ess']:.1f}/{t['minimum_weighted_tail_ess']:.1f}",
              f"Mean pixel MCSE on 0-255 scale: p99={t['pixel_mcse_p99']:.7g}; max={t['pixel_mcse_max']:.7g}"]
    for s in t['scores']:
        lines += [f"  {s['name']}: scaled QEP score={s['scaled_qep_score']:+.8g}; "
                  f"scaled MCSE={s['scaled_mcse']:.8g}; difference/MCSE={s['signed_difference_over_mcse']:+.6f}",
                  f"    MCSE(raw,fixed,weighted)=({s['raw_mcse']:.7g},{s['fixed_mcse']:.7g},{s['mcse']:.7g}); "
                  f"weighted/raw={s['weighted_to_raw_mcse']:.5f}; weighted/fixed={s['weighted_to_fixed_mcse']:.5f}",
                  f"    Four chain differences in raw-score units: {s['chain_difference']}",
                  f"    Screen crossed={s['descriptive_direction_resolved']}; all signs agree="
                  f"{s['all_four_chains_same_sign']}; candidate={s['candidate_direction']}"]
    lines += ['']
lines += ['INTERPRETATION', a['claim_scope']]
if decision == 'STOP_AUTOMATIC_EXPANSION_OF_THIS_LOCAL_GRADIENT_TRAINING_ROUTE':
    lines += ['The estimator precision repair replicated, but the declared coordinatewise',
              'gradient screen produced no eligible learning direction. Per the frozen',
              'stopping rule, do not automatically launch full QEP training, extend these',
              'chains, retry seeds or relax thresholds. This branch lacks support at the',
              'tested points and precision. Other optima or model structures were not tested.']
elif decision == 'BOUNDED_UPDATE_DESIGN_MAY_BE_CONSIDERED':
    lines += ['At least one direction met the frozen numerical screen after the precision',
              'repair. This motivates designing a bounded update, not an immediate claim',
              'of QEP evidence ascent, better boundaries or segmentation superiority.']
else:
    lines += ['The full validation criteria were not satisfied. Inspect target-level gates',
              'before interpreting gradient estimates; do not treat a numerical or precision',
              'failure as evidence that the QEP model is invalid.']
lines += ['No new AP metrics were produced. Earlier fixed-parameter end-to-end ties remain',
          'unchanged. The logged nuisance ECM candidate was not applied.', '',
          'REPRODUCTION',
          'source/ contains the exact prior base execution ZIP and offline/validation ZIP.',
          'Extract both source ZIPs into sibling folders, then run the validation script',
          'with --bundle pointing to the extracted base and --run only if rerunning is wanted.',
          'Keep existing outputs; the runner refuses to overwrite or automatically retry.',
          'The raw validation output is preserved under results/fresh_validation/.',
          'Its summaries, all score traces, saved states, estimates and hashes are included.',
          'summarize_validation.py audits those outputs against the frozen calibration bundle.', '']
(HERE/'REPORT_EN.txt').write_text('\n'.join(lines))

if len(a['targets']) == 2:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    labels = ['Mean', 'Log signal variance', 'Log noise ratio', 'Log beta rows', 'Log beta columns']
    maximum = max(4.5, 1.5+max(abs(s['signed_difference_over_mcse']) for t in a['targets'] for s in t['scores']))
    for ax, t in zip(axes, a['targets']):
        x = np.array([s['signed_difference_over_mcse'] for s in t['scores']])
        y = np.arange(5)
        ax.axvspan(-3, 3, color='#eef3f7')
        ax.axvline(0, color='#555555', linewidth=.8)
        for v in (-3, 3): ax.axvline(v, color='#b25535', linestyle='--', linewidth=1.2)
        ax.errorbar(x, y, xerr=1., fmt='o', color='#246da8', capsize=4, markersize=6)
        ax.set_xlim(-maximum, maximum)
        ax.set_yticks(y, labels)
        ax.set_title('Nuclei' if t['dataset'] == 'nuclei' else 'Whole-cell', fontweight='bold')
        ax.set_xlabel('(QEP score - analytic GP score) / MCSE')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#dddddd', linewidth=.6)
    axes[0].invert_yaxis()
    fig.suptitle('Fast-QEP: fresh-sample gradient validation', fontsize=16, fontweight='bold', y=.98)
    fig.text(.5, .90, 'Fixed weights; q = 1.5; same refitted GP parameter points', ha='center', fontsize=11)
    fig.text(.5, .025, 'Bars show one Monte Carlo standard error, not a confidence interval.\n'
                      'Dashed lines: descriptive 3-MCSE screen; size-floor and four-chain sign checks also apply.\n'
                      'New simulation seeds; the microscopy images are the same development images.',
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.18, right=.98, bottom=.23, top=.80, wspace=.18)
    fig.savefig(HERE/'fresh_gradient_validation.png', dpi=180)
    plt.close(fig)
print('Report generated from audited results.')
