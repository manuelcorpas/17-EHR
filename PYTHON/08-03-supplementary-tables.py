"""Supplementary Information for NCOMMS-26-040266-T revision 2: Tables S1 (scores before and after
correction) and S2 (publication and trial counts before and after), rendered from the sensitivity CSVs
to Markdown and, via pandoc, to docx. Usage: python PYTHON/08-03-supplementary-tables.py
"""
import copy, re, subprocess
from pathlib import Path
import docx
import pandas as pd
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

BASE = Path(__file__).resolve().parents[1]
SENS = BASE / 'ANALYSIS' / '03-11-NATCOMMS-R2-SENSITIVITY'
OUT = BASE / 'DOCX' / 'NATURE-COMMUNICATIONS-TRANSFER' / 'R2-REVISION'


FLOAT_COLS = {'Gap Score (submitted)', 'Gap Score (corrected)', 'Unified Score (submitted)', 'Unified Score (corrected)', 'DALYs',
              'Trial intensity (submitted)', 'Trial intensity (corrected)'}   # measured quantities keep one decimal


def md_table(df, cols, rename):
    d = df[cols].rename(columns=rename).copy()
    for c in d.columns:
        if d[c].dtype.kind in 'iu': d[c] = d[c].map(lambda x: f'{x:,}')
        elif d[c].dtype.kind == 'f' and d[c].dropna().apply(float.is_integer).all() and c not in FLOAT_COLS:
            d[c] = d[c].map(lambda x: '' if pd.isna(x) else f'{int(x):,}')
        elif c == 'DALYs': d[c] = d[c].map(lambda x: '' if pd.isna(x) else f'{x:,.0f}')   # whole DALYs; exact values in Source Data
        elif d[c].dtype.kind == 'f': d[c] = d[c].map(lambda x: '' if pd.isna(x) else (f'{x:.2f}' if abs(x) < 1 else f'{x:,.1f}'))
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


NUM = re.compile(r'^-?[\d,]+(\.\d+)?$')
# column widths in inches for a 7.0 in text block (Letter portrait, 0.75 in margins)
WIDTHS = {3: [1.6, 2.7, 2.7], 11: [1.45, 0.85, 0.85, 0.42, 0.42, 0.47, 0.47, 0.42, 0.42, 0.72, 0.5], 7: [1.9] + [0.85] * 6}
GROUPS = {11: [('Condition', 1), ('Category', 2), ('Gap Score', 2), ('Unified Score', 2), ('Rank', 2), ('DALYs', 1), ('Low burden', 1)],
          7: [('Condition', 1), ('Biobank publications', 2), ('Trials', 2), ('Trial intensity', 2)]}


def _set_cell(cell, text, size, bold=False, align=None, shade=None):
    for p in cell.paragraphs[1:]: p._p.getparent().remove(p._p)
    p = cell.paragraphs[0]
    for r in list(p.runs): r._r.getparent().remove(r._r)
    run = p.add_run(text); run.font.size = Pt(size); run.bold = bold
    mark = p._p.get_or_add_pPr().get_or_add_rPr() if hasattr(p._p.get_or_add_pPr(), 'get_or_add_rPr') else None
    if mark is None:
        pPr = p._p.get_or_add_pPr(); mark = pPr.find(qn('w:rPr'))
        if mark is None: mark = OxmlElement('w:rPr'); pPr.append(mark)
    for tag in ('w:sz', 'w:szCs'):
        e = mark.find(qn(tag))
        if e is None: e = OxmlElement(tag); mark.append(e)
        e.set(qn('w:val'), str(size * 2))
    p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(0)
    if align is not None: p.alignment = align
    if shade:
        tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement('w:shd')
        sh.set(qn('w:val'), 'clear'); sh.set(qn('w:color'), 'auto'); sh.set(qn('w:fill'), shade); tcPr.append(sh)


def _repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    for e in trPr.findall(qn('w:tblHeader')): trPr.remove(e)
    h = OxmlElement('w:tblHeader'); h.set(qn('w:val'), 'true'); trPr.append(h)


def format_docx(path):
    """Lay the pandoc output out for reading: portrait Letter, 0.75 in margins, fixed column widths with the condition
    column widest, 8 pt table text, a two-row header grouping submitted and corrected values, repeated on every page."""
    doc = docx.Document(str(path))
    for s in doc.sections:
        s.orientation = WD_ORIENT.PORTRAIT; s.page_width, s.page_height = Inches(8.5), Inches(11)
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(0.75)
    for t in doc.tables:
        ncol = len(t.columns); widths = WIDTHS[ncol]; size = 9 if ncol == 3 else 8
        tblPr = t._tbl.tblPr
        for tag in ('w:tblLayout', 'w:tblBorders', 'w:tblCellMar'):
            for e in tblPr.findall(qn(tag)): tblPr.remove(e)
        lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); tblPr.append(lay)
        tblW = tblPr.find(qn('w:tblW')); tblW.set(qn('w:type'), 'dxa'); tblW.set(qn('w:w'), str(round(sum(widths) * 1440)))
        b = OxmlElement('w:tblBorders')
        for edge in ('top', 'bottom', 'insideH'):
            e = OxmlElement(f'w:{edge}'); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '4'); e.set(qn('w:color'), 'BFBFBF'); b.append(e)
        tblPr.append(b)
        mar = OxmlElement('w:tblCellMar')
        for side in ('left', 'right'):
            e = OxmlElement(f'w:{side}'); e.set(qn('w:w'), '57'); e.set(qn('w:type'), 'dxa'); mar.append(e)
        tblPr.append(mar)
        for gc, w in zip(t._tbl.tblGrid.findall(qn('w:gridCol')), widths): gc.set(qn('w:w'), str(round(w * 1440)))
        header = t.rows[0]
        if ncol in GROUPS:                                     # two-row header: group name over Submitted | Corrected
            new = copy.deepcopy(header._tr); header._tr.addprevious(new)
            top, sub = t.rows[0], t.rows[1]
            for i, c in enumerate(sub.cells):
                txt = c.text
                _set_cell(c, 'Before' if txt.endswith('(submitted)') else 'After' if txt.endswith('(corrected)') else '',
                          size, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, shade='F2F2F2')
            col = 0
            for name, span in GROUPS[ncol]:                  # no vertical merges: they break the repeated header on later pages
                if span > 1:
                    cell = top.cells[col].merge(top.cells[col + span - 1])
                    _set_cell(cell, name, size, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, shade='F2F2F2')
                else:
                    _set_cell(top.cells[col], '', size, shade='F2F2F2')
                    _set_cell(sub.cells[col], name, size, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT if col == 0 else WD_ALIGN_PARAGRAPH.CENTER, shade='F2F2F2')
                col += span
            _repeat_header(top); _repeat_header(sub); first = 2
        else:
            for c in header.cells: _set_cell(c, c.text, size, bold=True, shade='F2F2F2')
            _repeat_header(header); first = 1
        body = [[c.text for c in r.cells] for r in t.rows[first:]]
        kind = []                                              # align whole columns: numbers right, flags centred
        for i in range(ncol):
            vals = [row[i] for row in body if row[i]]
            kind.append(WD_ALIGN_PARAGRAPH.RIGHT if i and vals and all(NUM.match(v) for v in vals) else
                        WD_ALIGN_PARAGRAPH.CENTER if vals and set(vals) <= {'True', 'False'} else None)
        for r in t.rows[first:]:
            for i, c in enumerate(r.cells): _set_cell(c, c.text, size, align=kind[i])
        for r in t.rows:
            for c, w in zip(r.cells, widths): c.width = Inches(w)
    doc.save(str(path))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = build(); (OUT / '06-Supplementary-Information-R2.md').write_text(md)
    subprocess.run(['pandoc', str(OUT / '06-Supplementary-Information-R2.md'), '-o', str(OUT / '06-Supplementary-Information-R2.docx')], check=True)
    format_docx(OUT / '06-Supplementary-Information-R2.docx')
    print('wrote', OUT / '06-Supplementary-Information-R2.docx')


if __name__ == '__main__':
    main()
