# QEP reconstruction × downstream: completed development experiment



**Verdict: no tested QEP pipeline surpassed the original FastGP pipeline on AP@0.5.** All four Q15 candidates fail the predeclared AP50 continuation requirements; this conclusion does not depend on the conservative auxiliary safety checks.



Scope: nuclei_figure_1 and whole_cell_figure_1 only. Two development images, not independent validation or a significance study. P = real R RobustGaSP + EBImage + paper cleanup. C = global Li + explicit peak markers + skimage watershed + cleanup.



## Primary results


| Dataset | Reconstruction | Downstream | AP50 | AP75 | FG Dice | Merged | Split |
|---|---|---|---:|---:|---:|---:|---:|
| nuclei | raw | P | 0.401473 | 0.147813 | 0.761948 | 15 | 8 |
| nuclei | raw | C | 0.693333 | 0.356838 | 0.839310 | 27 | 0 |
| nuclei | fastgp | P | 0.648101 | 0.284024 | 0.850393 | 22 | 1 |
| nuclei | fastgp | C | 0.620155 | 0.314465 | 0.862481 | 33 | 3 |
| nuclei | q2 | P | 0.460000 | 0.164894 | 0.783736 | 33 | 9 |
| nuclei | q2 | C | 0.625000 | 0.297297 | 0.856865 | 38 | 2 |
| nuclei | q15 | P | 0.539759 | 0.185529 | 0.802663 | 30 | 5 |
| nuclei | q15 | C | 0.598465 | 0.296680 | 0.855843 | 39 | 5 |
| whole_cell | raw | P | 0.429119 | 0.130303 | 0.814065 | 84 | 20 |
| whole_cell | raw | C | 0.527607 | 0.079480 | 0.827496 | 67 | 15 |
| whole_cell | fastgp | P | 0.583864 | 0.171115 | 0.858411 | 68 | 12 |
| whole_cell | fastgp | C | 0.527607 | 0.070201 | 0.823097 | 67 | 19 |
| whole_cell | q2 | P | 0.491968 | 0.171924 | 0.855048 | 74 | 18 |
| whole_cell | q2 | C | 0.445298 | 0.035763 | 0.788088 | 62 | 18 |
| whole_cell | q15 | P | 0.490835 | 0.165605 | 0.852405 | 73 | 10 |
| whole_cell | q15 | C | 0.450292 | 0.070504 | 0.808653 | 67 | 16 |



## Established findings



1. The completed run contains all 16 unique combinations. Independent review recomputed mask-level AP50/AP75, TP/FP/FN, foreground metrics, component counts and all four failure counts; all agree within 1e-12. Ten historical anchors agree with the frozen Round-5/Round-6 records. Every P/C pair shares the same independently verified reconstruction hash. Per-GT newly split and lost-match lists also match independent calculations. See INDEPENDENT_REVIEW.json.

2. Best Q15 AP50 among the two prespecified downstreams is 0.598465 (C) for nuclei versus FastGP-P 0.648101: difference -0.049636. Whole-cell best Q15 is 0.490835 (P) versus FastGP-P 0.583864: difference -0.093029. Choosing these best entries is a descriptive comparison across the two fixed options, not new validation.

3. A genuine within-pilot positive contrast is retained: nuclei Q15-P versus Q2-P gains AP50 +0.079759 and AP75 +0.020635; merged predictions fall 33 to 30 and split GT cells 9 to 5. Under C, the same nuclei q contrast reverses (AP50 -0.026535). Whole-cell q contrasts are -0.001133 under P and +0.004995 under C. Thus a consistent q-specific advantage is not supported.

4. P is not uniformly better for QEP. Moving C to P reduces nuclei Q15 AP50 from 0.598465 to 0.539759, while increasing whole-cell Q15 from 0.450292 to 0.490835. Neither beats the original FastGP-P anchor.

5. AP75 is not uniformly unfavorable to QEP: nuclei Q15-C exceeds FastGP-P by +0.012657 despite its lower AP50. Whole-cell Q2-P AP75 is 0.171924 versus FastGP-P 0.171115 (the same 109 TP75, with fewer FP75), while its AP50 is much lower. These limited metric trade-offs do not satisfy the primary superiority criterion.

6. Comparing the best-AP50 Q15 option with FastGP-P, nuclei merges increase 22 to 39, and whole-cell merges 68 to 73. The corresponding losses of previously matched GT cells are 43 and 46. These are descriptive overlapping error diagnostics, not proof of a single causal failure mechanism.

7. Whole-cell Q15-P has fewer total split GT cells than FastGP-P (10 versus 12), yet creates five newly split GT identities (5/403 = 1.24%). Net split count alone would hide this damage. Nuclei Q15-C creates five new splits (5/330 = 1.52%).



## Correct comparison to historical numbers



The nuclei FastGP-C value here is 0.620155, not the old 0.645669. This experiment deliberately supplies both P and C the identical R-side FastGP reconstruction, including original tiling/remainder handling. Round 5 and the reused C cell in Round 6 used a different Python reconstruction. The difference is expected provenance, not a failed anchor or a revision of the historical table. FastGP-P still reproduces the historical published-pipeline anchor.



## Interpretation and limits



The missing combination hypothesis—historical QEP reconstruction might become a superior full pipeline when given the real paper downstream—is not supported by these two fixed downstreams on these development images. This does not rule out different QEP models or future instance-separation methods. Q15/Q2 compare independently fitted models, including q-dependent likelihood behavior; the experiment does not establish a direct fixed-hyperparameter q effect in shallow predictive means.

No claim of out-of-sample generalization, statistical significance, global method optimality, or theoretical refutation is justified. AP means the frozen evaluator TP/(TP+FP+FN), with its historical row-wise IoU matching convention; it is not PR-curve area. Merge/split categories are nonexclusive.



## Reproduction incident retained



The first new nuclei Q2 reconstruction produced AP50 0.597436 rather than the historical 0.625 and triggered the intended stop. Original Round-3 and Round-5 cached reconstructions were then recovered from the user MacBook; all four QEP array pairs were identical. The completed experiment uses these verified historical arrays, as prioritized by the approved design. No anchor tolerance or model hyperparameter was relaxed.

The regeneration discrepancy remains unresolved. Aggregate fitted parameters and loss agree to floating-point precision, but reconstructed means differ (MAE 0.086801; maximum 0.951355 raw intensity units). Prediction computation is a lead, not an established cause. Successful cache-based downstream reproduction does not establish end-to-end training reproducibility. Preserve the incident and resolve it before presenting a fully reproducible training claim.



## Runtime and artifact provenance



QEP reconstruction/total runtimes are unavailable in the completed cached run, not zero. Original cache full-pipeline time is stored separately. FastGP reconstruction restores archived parameters and excludes historical optimizer time. No full-fit speed comparison is made.

Execution code: 2800fcafbf020b6f238109c2c1b21259c7325b23; authoritative pre-experiment checkpoint: fa746853ba007552bb1d5a43340d91f6796096b1; R reference: 44714c2e0be958fe796a8fd4bdbc220dae3c23dd. Arrays remain in the supplied QEP_4x2_results.zip and historical-cache archive; the GitHub result directory preserves all supplied text outputs and an array/file SHA256 manifest. It is not a self-contained replacement for those binary archives.



## Decision



Close this fixed-reconstruction × existing-downstream comparison and retain the negative full-pipeline result plus the conditional nuclei q contrast. Do not append q, threshold or architecture searches to turn the result positive. The main application lead remains instance separation, but this experiment did not test a new boundary-aware method. First document/resolve the prediction reproducibility gap; any subsequent boundary-aware experiment should have a separately frozen design and explicit approval. No additional experiment has been launched.
