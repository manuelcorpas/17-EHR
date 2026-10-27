"""Supplementary table: every condition's category, Gap and Unified score and rank before/after the 2026-10-25 fix."""
import json, os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
ARC = os.path.join(ROOT, 'ARCHIVE', '2026-10-25-PRE-CATEGORY-FIX', 'docs-data'); NOW = os.path.join(ROOT, 'docs', 'data')
INJ = {'Conflict_and_terrorism','Animal_contact','Drowning','Road_injuries','Self-harm','Interpersonal_violence','Falls',
 'Fire,_heat,_and_hot_substances','Poisonings','Exposure_to_mechanical_forces','Foreign_body','Environmental_heat_and_cold_exposure',
 'Other_transport_injuries','Executions_and_police_conflict','Other_unintentional_injuries','Adverse_effects_of_medical_treatment',
 'Paralytic_ileus_and_intestinal_obstruction'}
def frame(d, tag):
    i = pd.DataFrame(json.load(open(os.path.join(d, 'integrated.json')))['diseases'])
    c = pd.DataFrame([{'disease': x['id'].replace(' ', '_'), 'category': x['category']} for x in json.load(open(os.path.join(d, 'diseases.json')))['diseases']])
    i = i.merge(c, on='disease', how='left')
    i = i[~i.disease.isin(INJ) & i.unified_score.notna()].copy()
    i['rank'] = i.unified_score.rank(ascending=False, method='min').astype(int)
    keep = ['disease', 'category', 'gap_score', 'unified_score', 'rank'] + (['dalys', 'low_measured_burden'] if tag == 'corrected' else [])
    return i[keep].rename(columns={k: f'{k}_{tag}' for k in ['category', 'gap_score', 'unified_score', 'rank']})
def main():
    t = frame(ARC, 'published').merge(frame(NOW, 'corrected'), on='disease', how='outer')
    t['category_changed'] = t.category_published != t.category_corrected
    t['rank_change'] = t.rank_published - t.rank_corrected
    t = t.sort_values('rank_corrected')
    t['disease'] = t.disease.str.replace('_', ' ')
    t.round(2).to_csv(os.path.join(HERE, '08-supp-table-category-correction.csv'), index=False)
    print(len(t), 'rows;', int(t.category_changed.sum()), 'category changes;', int((t.gap_score_published != t.gap_score_corrected).sum()), 'gap changes')
    # Supplementary Table SY: biobank publication counts and trial counts per condition, published vs corrected
    dp = {d['id']: d for d in json.load(open(os.path.join(ARC, 'diseases.json')))['diseases']}
    dn = {d['id']: d for d in json.load(open(os.path.join(NOW, 'diseases.json')))['diseases']}
    cp = {d['name']: d for d in json.load(open(os.path.join(ARC, 'clinical_trials.json')))['diseases']}
    cn = {d['name']: d for d in json.load(open(os.path.join(NOW, 'clinical_trials.json')))['diseases']}
    rows = []
    for k in dn:
        rows.append({'condition': k, 'biobank_publications_published': dp.get(k, {}).get('research', {}).get('globalPublications'),
                     'biobank_publications_corrected': dn[k]['research']['globalPublications'],
                     'trials_published': cp.get(k, {}).get('trials'), 'trials_corrected': cn.get(k, {}).get('trials'),
                     'trial_intensity_published': cp.get(k, {}).get('intensity'), 'trial_intensity_corrected': cn.get(k, {}).get('intensity')})
    sy = pd.DataFrame(rows).sort_values('condition'); sy.round(2).to_csv(os.path.join(HERE, '15-supp-table-mapping-counts.csv'), index=False)
    print('SY rows', len(sy), 'pub count changed', int((sy.biobank_publications_published != sy.biobank_publications_corrected).sum()),
          'trial count changed', int((sy.trials_published.fillna(-1) != sy.trials_corrected.fillna(-1)).sum()))
if __name__ == '__main__': main()
