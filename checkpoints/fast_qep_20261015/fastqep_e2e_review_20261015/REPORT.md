# Fast-QEP original-pipeline replacement — development images

Actual R downstream executed. All 25 original tiles passed frozen numerical gates.

| Dataset | Arm | AP50 | AP75 | Merge | Split | Missed | Dice |
|---|---|---:|---:|---:|---:|---:|---:|
| nuclei | raw | 0.401473 | 0.147813 | 15 | 8 | 112 | 0.761948 |
| nuclei | fastgp | 0.648101 | 0.284024 | 22 | 1 | 74 | 0.850393 |
| nuclei | fastqep_q15 | 0.648101 | 0.284024 | 22 | 1 | 74 | 0.850295 |
| nuclei | gp_eta | 0.648101 | 0.284024 | 22 | 1 | 74 | 0.850295 |
| nuclei | q15_chain1 | 0.648101 | 0.284024 | 22 | 1 | 74 | 0.850295 |
| nuclei | q15_chain2 | 0.648101 | 0.284024 | 22 | 1 | 74 | 0.850295 |
| nuclei | q15_chain3 | 0.655612 | 0.300601 | 23 | 1 | 73 | 0.855961 |
| nuclei | q15_chain4 | 0.648101 | 0.284024 | 22 | 1 | 74 | 0.850295 |
| whole_cell | raw | 0.429119 | 0.130303 | 84 | 20 | 179 | 0.814065 |
| whole_cell | fastgp | 0.583864 | 0.171115 | 68 | 12 | 128 | 0.858411 |
| whole_cell | fastqep_q15 | 0.583864 | 0.171115 | 68 | 12 | 128 | 0.858395 |
| whole_cell | gp_eta | 0.583864 | 0.171115 | 68 | 12 | 128 | 0.858405 |
| whole_cell | q15_chain1 | 0.583864 | 0.171115 | 68 | 12 | 128 | 0.858405 |
| whole_cell | q15_chain2 | 0.583864 | 0.171115 | 68 | 12 | 128 | 0.858395 |
| whole_cell | q15_chain3 | 0.583864 | 0.171115 | 68 | 12 | 128 | 0.858400 |
| whole_cell | q15_chain4 | 0.583864 | 0.171115 | 68 | 12 | 128 | 0.858395 |

AP is the frozen legacy TP/(TP+FP+FN) score, not area under a precision-recall curve.
Chain rows diagnose Monte Carlo stability; they are not independent image replicates.
gp_eta is a fixed Gaussian smoothing-strength control, not an optimal fitted GP.
See comparisons.json for gained/lost GT identities, harm and descriptive support flags;
MC_stability.json for all four chain gains, and tile summaries for inference diagnostics.
One development image per dataset: no significance or generalization claim.
No new q search, downstream tuning, or held-out evaluation was performed.
