"""Supplementary Information for NCOMMS-26-040266-T revision 2: Tables S1 (scores before and after
correction) and S2 (publication and trial counts before and after), rendered from the sensitivity CSVs
to Markdown and, via pandoc, to docx. Usage: python PYTHON/08-03-supplementary-tables.py
"""
import subprocess
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SENS = BASE / 'ANALYSIS' / '03-11-NATCOMMS-R2-SENSITIVITY'
OUT = BASE / 'DOCX' / 'NATURE-COMMUNICATIONS-TRANSFER' / 'R2-REVISION'


def md_table(df, cols, rename):
    d = df[cols].rename(columns=rename).copy()
    for c in d.columns:
        if d[c].dtype.kind == 'f': d[c] = d[c].map(lambda x: '' if pd.isna(x) else (f'{x:.2f}' if abs(x) < 1 else f'{x:,.1f}'))
    d = d.fillna('')
    head = '| ' + ' | '.join(d.columns) + ' |\n|' + '|'.join(['---'] * len(d.columns)) + '|\n'
    return head + ''.join('| ' + ' | '.join(str(v) for v in row) + ' |\n' for row in d.itertuples(index=False))


def build():
    sx = pd.read_csv(SENS / '08-supp-table-category-correction.csv')
    sy = pd.read_csv(SENS / '15-supp-table-mapping-counts.csv')
    n_gap = int((sx.gap_score_published != sx.gap_score_corrected).sum()); n_cat = int(sx.category_changed.sum())
    n_pub = int((sy.biobank_publications_published != sy.biobank_publications_corrected).sum())
    n_tr = int((sy.trials_published.fillna(-1) != sy.trials_corrected.fillna(-1)).sum())
    md = f"""# Supplementary Information

Three Dimensions of Compounding Neglect: How Biobanks, Clinical Trials, and Scientific Literature Systematically Exclude the Global South

## Supplementary Note 1. Corrections made in this revision

The defects found in the analysis pipeline during the disease-mapping audit undertaken for this revision, and in the independent recomputation that followed it, are listed in the Methods (Corrections in this version). All were corrected before every result was recomputed from the raw inputs. Supplementary Table S1 lists, for every ranked condition, its gap-scoring category, Gap Score, Unified Neglect Score and rank in the previously submitted version and in the corrected version. Supplementary Table S2 lists the biobank publication count, registered trial count and trial intensity per condition in both versions. In total {n_cat} conditions changed category, {n_gap} changed Gap Score, {n_pub} changed biobank publication count and {n_tr} changed trial count.

### Hand-read precision of the corrected mapping

An analyst who had not seen the results reimplemented the matching rules from the raw MeSH terms and the curated term lists, reproduced the deposited mapping exactly (36,416 of 36,416 publications; 2,000 of 2,000 sampled trials), then read 20 randomly sampled linked records per condition (all records where fewer than 20). Counts are records judged to be about the condition.

| Condition | Biobank publications | Trials |
|---|---|---|
| Malaria | 13 of 17 | 20 of 20 |
| Breast cancer | 16 to 18 of 20 | 18 of 20 |
| Stroke | 12 of 20 clearly, 4 partly | not sampled |
| Maternal disorders | 9 of 20 (strict GBD definition); 13 of 20 (any pregnancy topic) | not sampled |
| Trachoma | 0 of 13 before the term-list correction (all genital Chlamydia trachomatis serology); 0 publications after | 16 to 17 of 20 |
| Neonatal disorders | not sampled | 11 of 20 (cystic fibrosis trials enter through the MeSH ancestor "Infant, Newborn, Diseases") |
| Lymphatic filariasis | not sampled | 11 of 20 (loiasis and onchocerciasis trials enter through "Filariasis") |
| Other neoplasms (residual category) | 0 of 20 as the GBD residual; 14 of 20 generic cancer records | not sampled |
| Other malignant neoplasms (residual category) | not sampled | 2 to 3 of 20 as the GBD residual before the residual-category correction |

## Supplementary Table S1. Scores and ranks per condition, previously submitted versus corrected

Conditions are ordered by corrected rank (1 = most neglected); injury and external causes are excluded from ranking. DALYs are from GBD 2021. The low-measured-burden flag marks conditions in the lowest decile of DALYs across all 175 GBD causes.

{md_table(sx, ['disease','category_published','category_corrected','gap_score_published','gap_score_corrected','unified_score_published','unified_score_corrected','rank_published','rank_corrected','dalys','low_measured_burden'],
          {'disease':'Condition','category_published':'Category (submitted)','category_corrected':'Category (corrected)','gap_score_published':'Gap Score (submitted)','gap_score_corrected':'Gap Score (corrected)','unified_score_published':'Unified Score (submitted)','unified_score_corrected':'Unified Score (corrected)','rank_published':'Rank (submitted)','rank_corrected':'Rank (corrected)','dalys':'DALYs','low_measured_burden':'Low measured burden'})}

## Supplementary Table S2. Biobank publication and trial counts per condition, previously submitted versus corrected

Biobank publications are peer-reviewed publications from the 70 IHCC cohorts mapped to the condition; trials are ClinicalTrials.gov registrations (2000-2025) mapped to the condition; trial intensity is trials per million DALYs. Conditions without trial data in the Translation dimension have blank trial columns.

{md_table(sy, ['condition','biobank_publications_published','biobank_publications_corrected','trials_published','trials_corrected','trial_intensity_published','trial_intensity_corrected'],
          {'condition':'Condition','biobank_publications_published':'Biobank publications (submitted)','biobank_publications_corrected':'Biobank publications (corrected)','trials_published':'Trials (submitted)','trials_corrected':'Trials (corrected)','trial_intensity_published':'Trial intensity (submitted)','trial_intensity_corrected':'Trial intensity (corrected)'})}
"""
    return md


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = build(); (OUT / '06-Supplementary-Information-R2.md').write_text(md)
    subprocess.run(['pandoc', str(OUT / '06-Supplementary-Information-R2.md'), '-o', str(OUT / '06-Supplementary-Information-R2.docx')], check=True)
    print('wrote', OUT / '06-Supplementary-Information-R2.docx')


if __name__ == '__main__':
    main()
