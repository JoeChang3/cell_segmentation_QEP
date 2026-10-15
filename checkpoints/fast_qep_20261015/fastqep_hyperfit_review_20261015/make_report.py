"""Render the completed mathematical review, not an experiment report."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'FastQEP_hyperparameter_feasibility_20261015.pdf'
DATA = json.loads((ROOT/'validation.json').read_text())
FONT = Path('/usr/share/fonts/truetype/dejavu')
pdfmetrics.registerFont(TTFont('DV', str(FONT/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DVB', str(FONT/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DVB')
NAVY = colors.HexColor('#15394F')
TEAL = colors.HexColor('#16777C')
LIGHT = colors.HexColor('#EEF5F6')
styles = {
 'title': ParagraphStyle('title', fontName='DVB', fontSize=22, leading=27, textColor=NAVY, spaceAfter=13),
 'h': ParagraphStyle('h', fontName='DVB', fontSize=12, leading=17, textColor=TEAL, spaceBefore=11, spaceAfter=5),
 'p': ParagraphStyle('p', fontName='DV', fontSize=9.2, leading=13.2, spaceAfter=7),
 'small': ParagraphStyle('small', fontName='DV', fontSize=8, leading=11, spaceAfter=6),
 'eq': ParagraphStyle('eq', fontName='DV', fontSize=9, leading=14, backColor=LIGHT, borderPadding=8, spaceBefore=5, spaceAfter=11),
 'cell': ParagraphStyle('cell', fontName='DV', fontSize=8.2, leading=11),
}
story=[]
def p(text, kind='p'):
    story.append(Paragraph(text, styles[kind]))
def h(text): p(text, 'h')
def eq(text): p(text, 'eq')
def table(rows, widths):
    t=Table([[Paragraph(str(x), styles['cell']) for x in row] for row in rows], colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LINEBELOW',(0,0),(-1,0),0.8,TEAL),('LINEBELOW',(0,1),(-1,-1),0.3,colors.HexColor('#D8E0E4')),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.extend([t, Spacer(1,8)])
def newpage(): story.append(PageBreak())

p('Fast-QEP: learning its own hyperparameters', 'title')
p('FEASIBILITY REVIEW · 15 OCTOBER 2026 · NO NEW CELL EXPERIMENT', 'small')
h('Decision: proceed to a bounded implementation pilot')
p('The current covariance-matched finite-tile Fast-QEP model admits a valid marginal-likelihood learning target. Its scores can be estimated from the existing type of posterior sampler. All 15 algebra and low-dimensional numerical checks passed. This establishes a usable learning mechanism, <b>not</b> a segmentation advantage, global optimizer, or large-image runtime guarantee.')
h('What the previous experiment actually established')
p('On one development image per dataset, q=1.5 reused the original R Fast-GP hyperparameters. Pooled Fast-QEP and Fast-GP had identical AP50: nuclei 0.6481012658; whole-cell 0.5838641189. This negative fixed-parameter result remains part of the record. It did not test QEP-specific hyperparameter estimation.')
p('The proposed hypothesis is narrower: learning the kernel ranges, noise ratio and nuisance parameters under the QEP model may change the reconstruction enough to help segmentation. This hypothesis remains untested on microscopy images.')
h('Completed numerical checks')
table([
 ['Check', 'Largest absolute discrepancy'],
 ['Complete-data score vs finite differences; q=2 and q=1.5', '5.61 × 10⁻⁹'],
 ['Kronecker vs dense score; 3 × 4 lattice', '2.46 × 10⁻¹³'],
 ['q=2 profiled evidence vs original R objective algebra', '3.56 × 10⁻¹⁵'],
 ['q=2 independent integral vs analytic GP evidence/moments/score', '4.45 × 10⁻¹⁶'],
 ['Evidence score vs finite differences of independent integration', '2.74 × 10⁻¹¹'],
 ['Evidence ascent and five nuisance-parameter ECM updates', 'Both q values increased their own objective'],
], [345, 155])
p('Checks use a fixed 12-dimensional lattice and a fixed two-observation quadrature fixture. They are identity checks, not a synthetic superiority benchmark. Parameters cannot be reliably identified from the two-observation fixture. Full results and tolerances are in validation.json.', 'small')
h('A necessary fairness correction')
p('The earlier frozen audit shows that the original R optimizer can stop on a flat, single-axis kernel plateau. Therefore a QEP improvement over the archived R result alone would not establish a QEP-specific benefit. A freshly fitted Fast-GP with the same fitting freedoms is mandatory.')

newpage()
p('1. Model and learning target', 'title')
p('Retain the finite-tile construction used in the latest Fast-QEP run. This review does not claim a new projectively consistent spatial process or silently replace it with independent coefficient-wise penalties.')
eq('y = m1 + v + ε;&nbsp;&nbsp; ε ~ N(0, sνI), independent of v<br/>'
   'v ~ qED<sub>n</sub>(0, C<sub>q</sub>);&nbsp;&nbsp; C<sub>q</sub> = sK / a(n,q)<br/>'
   'K = K<sub>col</sub> ⊗ K<sub>row</sub>;&nbsp;&nbsp; Cov(v) = sK<br/>'
   'a(n,q) = 2<super>2/q</super> Γ(n/2 + 2/q) / [n Γ(n/2)]')
p('Each factor is the original normalized-grid Matérn 5/2 correlation: k(d)=(1+x+x²/3)exp(−x), x=√5 βd. β is inverse range. Learn θ=(m, log s, log ν, log β<sub>row</sub>, log β<sub>col</sub>); q stays fixed at 1.5 or 2. Noise variance is sν, not ν.')
p('Let Q=vᵀK⁻¹v, r=aQ/s and α=n(q−2)/4. The normalized prior log density is:')
eq('log p(v|θ,q) = log(q/2) − (n/2)log(2π) − ½log|sK/a|<br/>'
   '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ αlog r − ½r<super>q/2</super>')
eq('ℓ<sub>q</sub>(θ) = log ∫ N(y; m1+v, sνI) p(v|θ,q) dv<br/>'
   '∇ℓ<sub>q</sub>(θ) = E<sub>v|y,θ,q</sub>[∇ log p(y,v|θ,q)]')
h('What must not be substituted')
p('The convolution with independent Gaussian noise is not automatically qED(y; m1, sK+sνI). The αlog r term and covariance factor a are required. Gaussian GLS and s=S/n are exact profiling formulas only at q=2; they can initialize q=1.5, but cannot be called its exact fitted solution.')
p('For q&lt;2 the physical-coordinate prior density is singular at v=0. Naively maximizing that density can prefer the singular point; this review uses integrated evidence and posterior expectations, not latent MAP. Mean control variates from the previous sampler must not be inserted as if they were raw posterior samples into quadratic or radial moments.')
h('q=2 limit')
p('At q=2, a=1 and α=0, so the prior is N(0,sK) and the data covariance is s(K+νI). Profiling m and s gives m=(1ᵀA⁻¹y)/(1ᵀA⁻¹1), S=(y−m1)ᵀA⁻¹(y−m1), s=S/n with A=K+νI. Restoring constants makes this exactly the R objective. This review checked the formula; it did not execute R.')

newpage()
p('2. Gradients and update rules', 'title')
p('The following scores hold the physical latent v fixed. They are not derivatives holding the whitened z fixed. Let e=y−m1−v, b=−n/2+eᵀe/(2sν), and K<sub>t</sub>=∂K/∂t for t=log β along either axis.')
eq('g<sub>m</sub> = 1ᵀe/(sν)<br/>'
   'g<sub>log ν</sub> = b<br/>'
   'g<sub>log s</sub> = (q/4)(r<super>q/2</super>−n) + b<br/>'
   'g<sub>t</sub> = −½tr(K⁻¹K<sub>t</sub>)<br/>'
   '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ [(q/4)r<super>q/2</super>−α] (vᵀK⁻¹K<sub>t</sub>K⁻¹v)/Q')
p('Average these quantities under the posterior to obtain the evidence gradient. In particular, the scale score includes the likelihood contribution b because changing s also changes the noise variance sν. Omitting b would optimize a different parameterization incorrectly.')
h('Nuisance-parameter ECM with fixed kernel')
p('Take expectations under the old posterior. Set m<sub>new</sub>=mean(y−E[v]). Define B=E[||y−m<sub>new</sub>1−v||²] and M=E[Q<super>q/2</super>]. At fixed old ν, solve the following scalar equation for s&gt;0:')
eq('(q/4)(a/s)<super>q/2</super>M + B/(2sν) − n(q+2)/4 = 0<br/>'
   'Then ν<sub>new</sub> = B/(ns<sub>new</sub>).')
p('The left side is strictly decreasing when the nonnegative moments are nonzero, giving a unique positive root. These are conditional maximizations of the expected complete log likelihood. Exact-expectation ECM is monotone; a finite-sample MC version needs error control and does not inherit monotonicity automatically. Recompute the posterior when the parameter target changes.')
h('Independent numerical oracle')
p('For n=2, integrate over t=R<super>q</super>/2~Exp(1) and a uniform angle. Adaptive radial integration with 128 angles was refined to 256. This provides an evidence oracle independent of the score formula and existing MCMC. Both evidence finite differences and the analytic Gaussian special case agree.')
p('One joint gradient step changed all four coordinates, including inverse range, and raised each model’s own evidence. Five ECM nuisance updates also raised each objective. These checks verify implementation identities; different objective values on this arbitrary fixture are not evidence that one model is better.')
h('Preserving the fast linear algebra')
p('The supplied Kronecker score computes with the two kernel factors, without constructing an n×n matrix. Kernel score evaluation still requires row/column matrix operations, not only elementwise radial updates. For r×c tiles its transforms scale as O(r²c+rc²); real-tile cost and numerical conditioning still need measurement.')

newpage()
p('3. Comparison design to preserve', 'title')
p('The fitting architecture should remain close to the original method: estimate shared (β<sub>row</sub>, β<sub>col</sub>, ν) from the first original tile, then reuse them on other tiles. Each tile gets its own m and s, fitted under the corresponding model. q=2 can profile these analytically; q=1.5 requires its own likelihood calculation. Do not add per-tile kernel fitting only to QEP.')
table([
 ['Arm', 'Role'],
 ['Archived original R Fast-GP', 'Historical pipeline reference; preserve its parameters and results.'],
 ['Refitted Fast-GP, q=2', 'Same model family, fitting data, parameter freedoms, initializations and convergence standards as the QEP arm. Analytic GP computations allowed.'],
 ['Refitted Fast-QEP, q=1.5', 'Its own evidence target and posterior mean; no q sweep.'],
 ['GP evaluated at QEP-fitted parameters', 'Tests whether any gain is reproducible by ordinary Gaussian reconstruction with the learned smoothing parameters.'],
 ], [190,310])
p('For mechanism attribution, add the complementary QEP-at-GP-fitted-parameters arm when proceeding beyond the feasibility gate. The resulting 2×2 crosses reconstruction distribution (GP/QEP) with parameter source (GP/QEP). Keep all nuisance parameters explicit during parameter swaps.')
h('The same downstream for the first comparison')
p('Use actual R RobustGaSP threshold selection, EBImage watershed and the same cleanup, image normalization, tile geometry, uncovered borders and instance metric. This isolates reconstruction/learning. Ground-truth masks must not select parameters, restarts, chains or downstream thresholds.')
h('Avoid optimizer-only wins')
p('The frozen September audit records nuclei R parameters β<sub>col</sub>≈5.68×10⁶ and whole-cell β<sub>row</sub>≈3.80×10⁷ on flat identity-kernel plateaus. Better likelihood basins were found without changing the GP model. The archived baseline remains valid as the original algorithm; a better estimator is a separate arm, not a silently corrected historical baseline.')
p('Prespecify an archived-parameter start plus one common nondegenerate start for both models. Bounds and any numerical regularization must be identical and documented. Final numeric bounds must wait for inspection of the recovered sampler’s conditioning safeguards. Do not invent them from unseen source or select them after looking at AP.')
h('Scope of conclusions')
p('Evidence improvement is not AP improvement. Lower merge counts are not sufficient if splits or lost correct instances rise. Beating the historical R algorithm but not refitted GP supports a pipeline/optimization improvement; it does not establish a QEP-specific segmentation advantage. Results on the two development images remain mechanism evidence, not generalization.')

newpage()
p('4. Gates, cost and reproducibility', 'title')
h('Completed now; not completed yet')
p('<b>Completed:</b> model derivation; raw-sample scores; Kronecker implementation; q=2 limit; independent quadrature and ECM checks. <b>Not completed:</b> integration with the prior sampler; MCMC score uncertainty on real tiles; fitted microscopy reconstructions; new AP or held-out evaluation. No Round 11 or other full-image run was launched.')
h('Bounded next step')
p('First integrate with the already validated sampler, then inspect only the two original first tiles. Begin with gradient estimates and their chain/MC uncertainty before any long optimization. Proposed initial checks are R-hat ≤1.01 and ESS ≥400 plus score-specific MC error assessment; retain the previous sampler’s additional safeguards after source recovery. Pixel-mean MC error alone is inadequate.')
p('Use the same two fixed initializations and a maximum of 8 accepted outer updates per initialization, with a 2-hour total pilot cap per dataset. These are resource limits, not convergence claims. Stop at the cap and report inconclusive if the fitting has not stabilized. Freeze numeric bounds, step acceptance and scaled-gradient tolerances after source integration but before running real-image fits; this package is not a ready-to-run real-data preregistration.')
p('Advance to whole-image reconstruction only if the target is implemented correctly, gradient noise permits resolved updates, no unexplained bound/conditioning failure occurs, and repeated inference supports stable reconstructed means. Compare reconstructions and parameter-source controls before spending time on downstream scoring. A pilot failure separates numerical infeasibility from evidence against the model.')
h('Resource estimate, not a benchmark')
p('The uploaded Mac run recorded 375.6 s for the first nuclei tile and 131.4 s for the first whole-cell tile; all 25 tiles used 109.6 min of sampling. Naively repeating a full 25-tile pass 20 times would cost about 36.5 h before kernel-score work and line searches. First-tile shared fitting reduces the initial scope; later per-tile nuisance fitting still costs work. Full-image training is therefore premature.')
h('Required handoff and rerun')
p('The current upload is a result-review bundle. It contains neither the old sampler source nor raw image/chain arrays. The earlier executable ZIP was not recovered in this environment. Supply FastQEP_R_pipeline_20261015.zip before integrating this review into that program. The new verify_hyperfit.py is a standalone derivation check, not recovered or modified production code.')
eq('python verify_hyperfit.py --output validation_local.json')
p('Requires NumPy and SciPy only. validation.json records versions, tolerances, fixture, seed and source SHA-256. This audit does not need R or microscopy data. requirements.txt records the tested versions; the file itself does not install packages.', 'small')
h('Sources and provenance')
p('[1] Li, O’Connor &amp; Lan (2023), Bayesian Learning via Q-Exponential Process, NeurIPS; density/whitening and §3.3.2. The evidence-score derivation here is an independent extension for the declared Gaussian-likelihood finite-tile model. <link href="https://proceedings.neurips.cc/paper_files/paper/2023/file/e6bfdd58f1326ff821a1b92743963bdf-Paper-Conference.pdf" color="#16777C">Official paper</link>.', 'small')
p('[2] Original R: UncertaintyQuantification/cell_segmentation, commit 44714c2e0be958fe796a8fd4bdbc220dae3c23dd, src/Modified_Functions_RGasp.R, lines 1-124. [3] JoeChang3/cell_segmentation_QEP, commit fa746853ba007552bb1d5a43340d91f6796096b1: TASK_A_FASTGP_BASELINE_AUDIT.md and PAPER_FAST_GP_WIRING_REPORT.md. [4] User-uploaded FastQEP_e2e_review_20261015.zip. Exact hashes/paths are in source_provenance.json.', 'small')

def footer(c, doc):
    c.setStrokeColor(colors.HexColor('#D8E0E4'))
    c.line(46,42,548,42)
    c.setFont('DV',7)
    c.setFillColor(NAVY)
    c.drawString(46,30,'FAST-QEP / HYPERPARAMETER FEASIBILITY / 2026-10-15')
    c.drawRightString(548,30,str(doc.page))

doc=SimpleDocTemplate(str(OUT),pagesize=(595.28,841.89),leftMargin=46,rightMargin=49,
                      topMargin=42,bottomMargin=54, title='Fast-QEP hyperparameter feasibility review',
                      author='Cell Segmentation Project')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
