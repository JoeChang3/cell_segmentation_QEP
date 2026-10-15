from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
A=json.loads((ROOT/'fastqep_audit_work/audit.json').read_text())
OUT=ROOT/'output/pdf/FastQEP_end_to_end_audit_20261015.pdf'
pdfmetrics.registerFont(TTFont('AuditSans','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('AuditSansBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('AuditSans',normal='AuditSans',bold='AuditSansBold',italic='AuditSans',boldItalic='AuditSansBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX',fontName='Helvetica-Bold',fontSize=20,leading=24,textColor=colors.HexColor('#17384A'),spaceAfter=9))
styles.add(ParagraphStyle(name='SubtitleX',fontSize=9,leading=13,textColor=colors.HexColor('#586B76'),spaceAfter=14))
styles.add(ParagraphStyle(name='BodyX',fontSize=10,leading=14,spaceAfter=8))
styles.add(ParagraphStyle(name='SmallX',fontSize=8.3,leading=11,spaceAfter=7))
styles.add(ParagraphStyle(name='LabelX',fontName='Helvetica-Bold',fontSize=11,leading=15,textColor=colors.HexColor('#17384A'),spaceBefore=10,spaceAfter=7))
for key in ['TitleX','SubtitleX','BodyX','SmallX','LabelX']:
    styles[key].fontName='AuditSansBold' if key in ['TitleX','LabelX'] else 'AuditSans'
styles['BodyX'].fontSize=9.5;styles['BodyX'].leading=13.4
styles['SmallX'].fontSize=8.1;styles['SmallX'].leading=10.8
story=[]
def para(text,style='BodyX'):story.append(Paragraph(text,styles[style]))
def tab(rows,widths):
    content=[[Paragraph(str(c),styles['SmallX']) for c in row] for row in rows]
    t=Table(content,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8F0F4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#A9BCC7')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D9E1E6')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(t);story.append(Spacer(1,8))

para('Fast-QEP: end-to-end result audit','TitleX')
para('Cell segmentation | Experiment ID 20261015 | Reviewed 14 Oct 2026, America/Phoenix','SubtitleX')
para('<b>Finding:</b> the prescribed q=1.5 posterior-mean substitution produced small reconstruction and foreground changes, but no AP50/AP75 or merge/split improvement on either development image. This is an application negative for this fixed configuration, not a refutation of QEP theory.')
para('Primary comparison: original R downstream held fixed','LabelX')
rr=[['Image / arm','AP50','AP75','Merge / split','FG Dice']]
for ds,label in [('nuclei','Nuclei'),('whole_cell','Whole-cell')]:
 for key,arm in [('baseline','Fast-GP'),('primary','Fast-QEP')]:
  d=A['datasets'][ds][key]
  rr.append([f'{label} / {arm}',f"{float(d['ap50']):.8f}",f"{float(d['ap75']):.8f}",f"{d['merged']} / {d['split']}",f"{float(d['fg_dice']):.8f}"])
tab(rr,[151,84,84,89,96])
para('AP here is the frozen legacy TP/(TP+FP+FN) score, with per-GT-best-IoU matching. It is not precision-recall curve area. Primary QEP is the pooled four-chain estimate; individual chains are diagnostics.','SmallX')
para('Why equal AP does not mean equal images','LabelX')
rr=[['Quantity','Nuclei','Whole-cell']]
rr += [['Mean absolute reconstruction difference<br/>(0-255 units; estimated QEP mean)',f"{A['datasets']['nuclei']['whole_image_mean_absolute_q_change_0_255']:.7f}",f"{A['datasets']['whole_cell']['whole_image_mean_absolute_q_change_0_255']:.7f}"],['Foreground pixels changed vs Fast-GP','121 / 1,085,136 (0.01115%)','7 / 361,200 (0.001938%)'],['Gained / lost matched GT at IoU 0.5','0 / 0','0 / 0'],['TP / FP / FN at IoU 0.5<br/>(same in both primary arms)','256 / 65 / 74','275 / 68 / 128']]
tab(rr,[254,125,125])
para('The mean differences are area-weighted from all tile summaries, including the unchanged two trailing columns. They include finite Monte Carlo error. Initial/final threshold proportions and outlier flags are unchanged in all 25 pooled-QEP tiles; intensity thresholds change slightly because tile maxima change. The new masks are not identical to Fast-GP masks, but no GT crosses the AP50 matching decision. Nuclei spurious count falls 23 to 22 while foreground Dice falls slightly; this is not a merge-reduction gain.','SmallX')
para('Inference and baseline checks','LabelX')
tab([['Recorded diagnostic','Nuclei (16 tiles)','Whole-cell (9 tiles)'],['Worst monitored Rhat','1.000925','1.000872'],['Minimum bulk / tail ESS','8089.9 / 7399.2','7812.5 / 5628.5'],['Largest pixel MCSE (0-255)','0.000659','0.002177'],['Sampling time only','88.62 min','20.95 min']],[254,125,125])
para('All 25 recorded gates and prescribed seed/geometry checks pass; no recorded retained divergence, nonfinite HMC proposal or global numerical failure. Total sampling time: 109.58 min. Actual R baseline records match historical masks/partitions; Python q=2 vs R maximum normalized error is 8.25e-15 / 1.13e-13. These are monitored diagnostics, not a mathematical proof of mixing.','SmallX')

story.append(PageBreak())
para('The exceptional chain and the decision','TitleX')
para('A threshold-selection sensitivity finding, not a selected positive QEP result','SubtitleX')
para('Nuclei chain 3 is the only chain with an AP gain','LabelX')
tab([['Quantity','Fast-GP / pooled QEP','QEP chain 3'],['AP50','0.64810127','0.65561224'],['AP75','0.28402367','0.30060120'],['Merge / split','22 / 1','23 / 1'],['Matched GT changes vs Fast-GP','Pooled: gain 0, lose 0','Gain 4, lose 3'],['Changed foreground vs Fast-GP','Pooled: 121 pixels','5,715 pixels'],['Tile 9 initial / final proportion','0.76 / 0.76','0.69 / 0.72'],['Tile 9 components before outlier repair<br/>(R count includes background)','30','199'],['Outlier tiles','Tile 5 only','Tiles 5 and 9']],[254,125,125])
para('The added outlier also changes tile 5\'s final proportion from 0.7226667 to 0.72. All outlier flags, replacement-proportion arithmetic and p x maximum intensity thresholds were checked across the 16 supplied threshold tables. Chain 3 gains GT labels 7, 9, 10 and 67, but loses 54, 61 and 86. Its maximum threshold-intensity shift vs Fast-GP is about 9.98 units, versus 0.0528 for pooled QEP.')
para('<b>Interpretation:</b> these records localize the exceptional result to threshold selection and outlier handling. They support sensitivity of the downstream decisions to chain-mean perturbations. The criterion landscape and masks were not supplied, so a controlled causal attribution is not established. Low pixel MCSE alone does not guarantee stability of discrete thresholding or AP. Three other nuclei chains tie the baseline; every whole-cell chain ties it. Choosing chain 3 would violate the frozen primary estimator.')
para('What the Gaussian control establishes','LabelX')
para('The fixed g_eta Gaussian smoothing-strength control ties QEP on AP50/AP75 and merge/split in both images; its nuclei endpoint metrics also coincide. This provides no QEP-specific advantage at these endpoints. It does not establish exact equality between GP and QEP posterior distributions or reconstruction arrays.')
para('Decision and limits','LabelX')
para('<b>Freeze this branch as a negative result:</b> fixed R hyperparameters + covariance-matched finite-tile qED prior + independent Gaussian noise + q=1.5 posterior mean + unchanged original R downstream did not improve the prescribed instance endpoints. Repeating this configuration or selecting q/seeds after observing AP has no current justification. If Fast-QEP work continues, first justify a structural mechanism that can add useful spatial separation evidence beyond a Gaussian smoothing-strength change. This audit starts no new experiment.')
para('<b>Scope:</b> one development image per dataset; no significance, equivalence-test or held-out generalization claim. The lightweight archive omits arrays, chains, source files and detailed R logs. This is a consistency audit of supplied records, not independent recomputation of masks, IoU or convergence statistics from raw draws. All 72 uploaded files covered by the completion manifest match their recorded hashes; 214 manifest-listed files were not supplied. Sixteen AP50/AP75 rows were recomputed from counts; runtime signatures and all 25 checkpoint summaries agree.','SmallX')
para('Sources: metrics.csv; comparisons.json; MC_stability.json; environment.json; completion.json; run.log; preflight/R_anchor_checks.json; preflight/preflight_complete.json; all uploaded tile summaries/checkpoints and threshold tables. Numeric audit details are attached to this PDF as audit.json.','SmallX')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#CBD7DE'));canvas.line(45,36,549,36)
    canvas.setFont('AuditSans',8);canvas.setFillColor(colors.HexColor('#586B76'))
    canvas.drawString(45,24,'Fast-QEP / original R pipeline - development-image evidence only')
    canvas.drawRightString(549,24,str(doc.page))

doc=SimpleDocTemplate(str(OUT),pagesize=(594,842),leftMargin=45,rightMargin=45,topMargin=39,bottomMargin=49)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(OUT);writer=PdfWriter();writer.clone_document_from_reader(reader)
writer.add_attachment('audit.json',(ROOT/'fastqep_audit_work/audit.json').read_bytes())
temp=OUT.with_suffix('.tmp.pdf')
with temp.open('wb') as f:writer.write(f)
temp.replace(OUT)
print(OUT,'pages=',len(PdfReader(OUT).pages))
