"""Source Data workbook for NCOMMS-26-040266-T: one sheet per display item, values as plotted or tabulated.
Reads docs/data/*.json, the sensitivity outputs, the ED Fig 1 source CSVs and the Fig 4a UMAP coordinates. Writes
ANALYSIS/08-NATCOMMS-R2-FIGURES/Source_Data.xlsx. Usage: python PYTHON/08-02-source-data.py
"""
import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DOCS = BASE / 'docs' / 'data'; SENS = BASE / 'ANALYSIS' / '03-11-NATCOMMS-R2-SENSITIVITY'; FIGS = BASE / 'ANALYSIS' / '08-NATCOMMS-R2-FIGURES'
INJ = {'Conflict_and_terrorism','Animal_contact','Drowning','Road_injuries','Self-harm','Interpersonal_violence','Falls',
       'Fire,_heat,_and_hot_substances','Poisonings','Exposure_to_mechanical_forces','Foreign_body','Environmental_heat_and_cold_exposure',
       'Other_transport_injuries','Executions_and_police_conflict','Other_unintentional_injuries','Adverse_effects_of_medical_treatment',
       'Paralytic_ileus_and_intestinal_obstruction'}


def sheets():
    L = lambda f: json.load(open(DOCS / f))
    B, D, C, S, I = L('biobanks.json')['biobanks'], L('diseases.json')['diseases'], L('clinical_trials.json'), L('semantic.json'), L('integrated.json')
    out = {}
    out['Fig2b_biobank_EAS'] = pd.DataFrame([{'biobank': b['name'], 'country': b.get('country'), 'EAS': b['scores']['equityAlignment'],
                                              'category': b['scores']['equityCategory'], 'publications': b['stats']['totalPublications'],
                                              'income_group': 'Global South' if b['isGlobalSouth'] else 'High-income', 'region': b['regionName']} for b in B])
    out['Fig2a_country_EAS'] = out['Fig2b_biobank_EAS'].groupby('country', dropna=True)['EAS'].mean().reset_index().rename(columns={'EAS': 'mean_EAS'})
    out['Fig2c_publication_share'] = out['Fig2b_biobank_EAS'].groupby('income_group')['publications'].sum().reset_index()
    geo = C['geographic']; out['Fig3a_trial_sites'] = pd.DataFrame(geo['topCountries']) if 'topCountries' in geo else pd.DataFrame([geo])
    out['Fig3b_GS_intensity'] = pd.DataFrame([C['globalSouthAnalysis']])
    gl = {d['id'].replace(' ', '_').replace('/', '_'): (d['category'], d['globalSouthPriority']) for d in D}
    sem = pd.DataFrame([{'disease': k, 'n_papers': v['n_papers'], 'SII': v['sii'], 'KTP': v['ktp'], 'RCC': v['rcc'],
                         'category': gl.get(k, (None, None))[0], 'global_south_priority': gl.get(k, (None, None))[1]} for k, v in S['metrics'].items()])
    out['Fig4b_e_semantic'] = sem.sort_values('SII', ascending=False)
    rows = [d for d in I['diseases'] if d['unified_score'] is not None and d['disease'] not in INJ]
    rows.sort(key=lambda d: -d['unified_score'])
    uni = pd.DataFrame(rows)[['disease', 'unified_score', 'gap_score', 'ct_equity', 'sii', 'dimensions_available', 'dalys', 'low_measured_burden']]
    uni.insert(0, 'rank', range(1, len(uni) + 1))
    out['Fig5_unified_top30'] = uni.head(30); out['Table1_top10'] = uni.head(10); out['Unified_all'] = uni
    for p in 'abc':
        f = FIGS / f'Extended_Data_Fig_01{p}_source.csv'
        if f.exists(): out[f'EDFig1{p}'] = pd.read_csv(f)
    out['EDFig2_sensitivity'] = pd.read_csv(SENS / '02-r2-sensitivity-scores.csv')
    out['EDFig3_regional'] = out['Fig2b_biobank_EAS'].groupby('region').agg(biobanks=('biobank', 'count'), publications=('publications', 'sum'), mean_EAS=('EAS', 'mean')).reset_index()
    out['SuppTable_S1_scores'] = pd.read_csv(SENS / '08-supp-table-category-correction.csv')
    out['SuppTable_S2_counts'] = pd.read_csv(SENS / '15-supp-table-mapping-counts.csv')
    w = I['weights']
    out['Fig1c_weights'] = pd.DataFrame({'dimension': ['Discovery', 'Translation', 'Knowledge'],
                                         'weight': [w['discovery'], w['translation'], w['knowledge']],
                                         'pc1_explained_variance': w['pc1_explained_variance'], 'n_complete_cases': w['n_complete_cases']})
    out['Fig4a_UMAP'] = pd.read_csv(FIGS / 'fig4a_umap_coords.csv')  # written by the UMAP run that drew Fig 4a (seed 42)
    return out


def main():
    sh = sheets(); path = FIGS / 'Source_Data.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        for name, df in sh.items(): df.to_excel(w, sheet_name=name[:31], index=False)
    print(path, {k: len(v) for k, v in sh.items()})


if __name__ == '__main__':
    main()
