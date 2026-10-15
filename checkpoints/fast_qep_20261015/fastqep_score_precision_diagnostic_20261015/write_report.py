"""Create reports and a scientific figure from the frozen offline analysis."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
a = json.loads((HERE/'results/analysis.json').read_text())
weights = json.loads((HERE/'candidate_weights.json').read_text())
labels = ['Mean', 'Log signal\nvariance', 'Log noise\nratio', 'Log beta\nrows', 'Log beta\ncolumns']
fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4))
colors = ['#246DA8', '#D87921']
for col, dataset in enumerate(('nuclei', 'whole_cell')):
    rows = [r for r in a['rows'] if r['dataset'] == dataset and r['parameter_source'] == 'refitted_GP']
    x = np.arange(5)
    for k, title in enumerate(('Chains 1-2 calibrate; 3-4 evaluate', 'Chains 3-4 calibrate; 1-2 evaluate')):
        offset = (k-.5)*.20
        axes[0, col].plot(x+offset, [r['splits'][k]['calibration']['weight'] for r in rows],
                          'o', ms=7, color=colors[k], label=title)
        axes[1, col].plot(x+offset, [r['splits'][k]['weighted_to_fixed_mcse'] for r in rows],
                          'o', ms=7, color=colors[k])
    for ax in axes[:, col]:
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylim(-.06, 1.20)
        ax.set_xlim(-.5, 4.5)
        ax.axhline(1, color='#777777', linewidth=1, linestyle='--')
        ax.grid(axis='y', color='#dddddd', linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, col].set_title('Nuclei: refitted GP point' if dataset == 'nuclei' else 'Whole-cell: refitted GP point', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Calibration-selected weight b\n(original correction: b = 1)')
axes[1, 0].set_ylabel('Evaluation MCSE / original MCSE\n(lower means more precise on these chains)')
fig.suptitle('Fast-QEP: offline score precision diagnostic', fontsize=16, fontweight='bold', y=.98)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .935), ncol=2, frameon=False, fontsize=10)
fig.text(.5, .03, 'Existing pilot samples; reciprocal splits are not independent replications.\n'
                 'No fresh sampling, QEP parameter update, or segmentation evaluation.',
         ha='center', va='bottom', fontsize=10, color='#444444')
fig.subplots_adjust(top=.84, bottom=.17, left=.10, right=.98, hspace=.42, wspace=.22)
fig.savefig(HERE/'precision_diagnostic.png', dpi=180)
plt.close(fig)

lines = ['FAST-QEP OFFLINE SCORE PRECISION DIAGNOSTIC', '2026-10-15', '',
         'Outcome: fixed-control overcorrection can be reduced, but no stable QEP gradient',
         'signal emerged from the separated-chain diagnostic.', '',
         'This is a post-hoc analysis of already inspected data, not fresh validation.',
         'Original pilot results and decisions remain unchanged. No model fitting, new',
         'sampling, QEP hyperparameter update, segmentation or AP evaluation was performed.', '',
         'METHOD',
         'The unchanged existing control is C=G_raw-G_fixed, with E(C)=0 under the target.',
         'Compare G_raw, G_fixed=G_raw-C, and G_raw-b*C.',
         'Fit b on centered calibration batch means (block length 20 score draws),',
         'using Cov(G,C)/Var(C) clipped to [0,1]. Evaluate on the other two chains.',
         'Report both reciprocal splits without pooling their MCSEs.',
         'The batch criterion is empirical; asymptotic optimality is not asserted.',
         'All MCSE comparisons below use the same evaluation chains.', '',
         'FINDINGS',
         '12 of 18 informative parameter/target combinations meet the fixed offline',
         'technical screen in both splits. Two archived kernel axes are identically zero.',
         'All weighted evaluation diagnostics pass the specified R-hat/ESS thresholds.',
         'No weighted score crosses the old descriptive 3-MCSE plus scaled-floor screen',
         'in either split. Maximum absolute difference/MCSE across all weighted outputs',
         'is 1.8966. This is not a p-value or an equivalence conclusion.', '',
         'The whole-cell kernel repair mostly returns precision to the raw-score level:',
         'weighted/raw MCSE is 0.972-0.990 for rows and 0.917-1.008 for columns at the',
         'refitted GP point. It does not create substantial extra information over raw scores.',
         'The nuclei kernel estimates improve more clearly over raw as well as fixed.', '',
         'WHY THE OLD CONTROL WORSENED PRECISION',
         'Var(G-bC)=Var(G)+b^2 Var(C)-2b Cov(G,C), also for batch means.',
         'A valid zero-mean control preserves the expectation but does not guarantee',
         'lower variance at b=1. The observed coefficient and covariance results show',
         'that b=1 was too large for these q=1.5 kernel scores, especially whole-cell.',
         'This diagnoses the estimator, not a failure of the QEP density or sampler.', '',
         'DETAILED EVALUATION (raw score units)',
         'Each block: b; MCSE(raw, fixed, weighted); weighted mean; QEP-GP / MCSE;',
         'and weighted convergence diagnostics. The sign is descriptive only.']
for r in a['rows']:
    lines += ['', f"{r['dataset']}/{r['parameter_source']}/{r['name']}"]
    if r['uninformative_zero']:
        lines += ['  Constant zero coordinate: no information; excluded from reduction counts.']
        continue
    for s in r['splits']:
        e = s['estimates']; w = e['weighted']
        lines += [f"  {s['split']}: b={s['calibration']['weight']:.8g}; MCSE="
                  f"({e['raw']['mcse']:.7g}, {e['fixed']['mcse']:.7g}, {w['mcse']:.7g}); "
                  f"mean={w['mean']:+.7g}; difference/MCSE={w['signed_difference_over_mcse']:+.5f}",
                  f"    MCSE/fixed={s['weighted_to_fixed_mcse']:.5f}; MCSE/raw={s['weighted_to_raw_mcse']:.5f}; "
                  f"Rhat={w['rhat_rank_folded_split']:.6f}; bulk/tail ESS="
                  f"{w['ess_bulk']:.1f}/{w['ess_tail']:.1f}; chain means={w['chain_means']}"]
    lines += ['  Technical candidate in both splits: '+str(r['technical_candidate_both_splits'])]
lines += ['', 'CANDIDATE FOR FUTURE VALIDATION, NOT VALIDATED',
          'The following coefficients use all four OLD chains under the same scalar rule.',
          'They are frozen for a possible later fresh-sample check; their effectiveness',
          'has not been independently established. They are not re-used to claim a new',
          'confirmatory result on these calibration data.',
          'Order: mean, log s, log nu, log beta rows, log beta columns.']
for t in weights['targets']:
    lines += [t['dataset']+': '+str(t['weights'])]
lines += ['', 'NEXT STEP AND STOPPING',
          'A separately approved fresh run can validate the corrected estimator on only',
          'the two refitted GP points, with new fixed seeds, 4x4000 retained samples,',
          'and 30-minute overall budget. Its runner is prepared and defaults to checks.',
          'Read NEXT_VALIDATION_PROTOCOL.txt for exact gates and stopping rules.',
          'No fresh validation was started during this task.',
          'If precision and convergence pass but directions remain unresolved, do not',
          'automatically expand training or sampling. That would constrain this local',
          'gradient-learning route, not mathematically refute QEP or all other basins.', '',
          'REPRODUCTION',
          'python analyze.py --out regenerated_analysis',
          'The package includes exact score arrays, summaries, unchanged diagnostics,',
          'hashes, code and protocols. It excludes unnecessary original latent states.',
          'validate_new_samples.py requires the prior FastQEP_hyperfit_pilot_20261015',
          'bundle; --bundle supplies its path. Without --run it performs checks only.', '',
          'BACKGROUND',
          'Dellaportas and Kontoyiannis (2012), JRSS B 74, 133-161:',
          'https://www.dpmms.cam.ac.uk/~ik355/PAPERS/cvmcmcJ.pdf',
          'Used for general control-variate/MCMC context. Its specialized reversible',
          'kernel estimator and optimality results are not claimed for our composed sampler.', '']
(HERE/'REPORT_EN.txt').write_text('\n'.join(lines))
print('Report and figure written.')
