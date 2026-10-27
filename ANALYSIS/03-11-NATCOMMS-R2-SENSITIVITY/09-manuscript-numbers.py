"""Every manuscript number that depends on Gap/Unified/EAS scores, computed from the archived
(pre-fix) data and from the corrected data. The archived column must reproduce the submitted text."""
import json, os
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
SRC = {'previous': os.path.join(ROOT, 'ARCHIVE', '2026-10-25-PRE-CATEGORY-FIX', 'docs-data'),
       'corrected': os.path.join(ROOT, 'docs', 'data')}
WHO = ['Chagas_disease','Dengue','Guinea_worm_disease','Cystic_echinococcosis','Food-borne_trematodiases',
       'African_trypanosomiasis','Leishmaniasis','Leprosy','Lymphatic_filariasis','Onchocerciasis','Rabies','Scabies',
       'Schistosomiasis','Cysticercosis','Trachoma']  # WHO NTD Roadmap 2021-2030 as listed in Methods; 15 map to GBD Level 3
# (06-10 used a different list incl. malaria, typhoid and yellow fever and omitting scabies and echinococcosis)
INJ = {'Conflict_and_terrorism','Animal_contact','Drowning','Road_injuries','Self-harm','Interpersonal_violence','Falls',
 'Fire,_heat,_and_hot_substances','Poisonings','Exposure_to_mechanical_forces','Foreign_body','Environmental_heat_and_cold_exposure',
 'Other_transport_injuries','Executions_and_police_conflict','Other_unintentional_injuries','Adverse_effects_of_medical_treatment',
 'Paralytic_ileus_and_intestinal_obstruction'}

def numbers(d):
    I = json.load(open(os.path.join(d, 'integrated.json'))); D = json.load(open(os.path.join(d, 'diseases.json')))['diseases']
    B = json.load(open(os.path.join(d, 'biobanks.json')))['biobanks']
    w = I['weights']; df = pd.DataFrame(I['diseases'])
    gsp = {x['id'].replace(' ', '_'): x['globalSouthPriority'] for x in D}
    u = df[df.unified_score.notna()]
    dis = u[~u.disease.isin(INJ)].sort_values('unified_score', ascending=False)
    top = dis.head(10)
    who = {x.lower() for x in WHO}; isw = lambda s: s.lower() in who
    # disease-level statistics exclude injury/external causes, as the ranking does (reproduces the submitted 27.7/22.2)
    ntd = dis[dis.disease.map(isw)].unified_score; non = dis[~dis.disease.map(isw)].unified_score
    g = pd.Series([x['gap']['score'] for x in D])
    eas = pd.DataFrame([{'name': x['name'], 'eas': x['scores']['equityAlignment'], 'cat': x['scores']['equityCategory']} for x in B])
    e = lambda n: float(eas.loc[eas.name.str.startswith(n), 'eas'].iloc[0])
    gs = df.dropna(subset=['gap_score', 'sii'])
    sc = lambda n: float(u.loc[u.disease == n, 'unified_score'].iloc[0])
    return {
        'w_D': w['discovery'], 'w_T': w['translation'], 'w_K': w['knowledge'], 'pc1_pct': round(100 * w['pc1_explained_variance'], 1),
        'w2_D': round(w['discovery'] / (w['discovery'] + w['knowledge']), 2), 'w2_K': round(w['knowledge'] / (w['discovery'] + w['knowledge']), 2),
        'unified_mean': round(dis.unified_score.mean(), 1), 'unified_sd': round(dis.unified_score.std(), 1),
        'unified_min': f"{dis.iloc[-1].disease} {dis.unified_score.min():.1f}",
        'unified_max': f"{dis.iloc[0].disease} {dis.unified_score.max():.1f}",
        'breast_cancer': round(sc('Breast_cancer'), 1),
        'top10_who_ntd_excl_malaria': int(sum(isw(x) and x != 'Malaria' for x in top.disease)),
        'top10_low_burden': int(top.get('low_measured_burden', pd.Series(False, index=top.index)).fillna(False).sum()),
        'top10': '; '.join(f"{r.disease} {r.unified_score:.1f}/gap {r.gap_score:.0f}" for r in top.itertuples()),
        'top10_gs': int(sum(gsp.get(x, False) for x in top.disease)), 'top10_who_ntd': int(sum(map(isw, top.disease))),
        'ntd_mean': round(ntd.mean(), 1), 'non_ntd_mean': round(non.mean(), 1),
        'ntd_p': f"{stats.mannwhitneyu(ntd, non, alternative='greater').pvalue:.1e}",
        'gap_critical': int((g > 70).sum()), 'gap_high': int(((g > 50) & (g <= 70)).sum()),
        'gap_moderate': int(((g > 30) & (g <= 50)).sum()), 'gap_low': int((g <= 30).sum()), 'gap_n': len(g),
        'eas_uk': e('UK Biobank'), 'eas_high': int((eas.cat == 'High').sum()), 'eas_mod': int((eas.cat == 'Moderate').sum()),
        'eas_low': int((eas.cat == 'Low').sum()), 'eas_low_pct': round(100 * (eas.cat == 'Low').mean()),
        'eas_nurses': e("Nurses"), 'eas_whi': e("Women"), 'eas_estonia': e("Estonian"),
        'eas_median': round(eas.eas.median(), 1), 'eas_q1': round(eas.eas.quantile(.25), 1), 'eas_q3': round(eas.eas.quantile(.75), 1),
        'r_gap_sii': round(stats.pearsonr(gs.gap_score, gs.sii)[0], 2),
        'ihd': round(sc('Ischemic_heart_disease'), 1), 'stroke': round(sc('Stroke'), 1),
        'n_who_mapped': int(sum(map(isw, dis.disease))),
        'malaria_rank': int(list(dis.disease).index('Malaria') + 1),
    }

def main():
    t = pd.DataFrame({k: numbers(v) for k, v in SRC.items()})
    t['changed'] = t.previous.astype(str) != t.corrected.astype(str)
    t.to_csv(os.path.join(HERE, '10-manuscript-numbers.csv'))
    pd.set_option('display.max_colwidth', 400); pd.set_option('display.width', 250)
    print(t.to_string())

if __name__ == '__main__': main()
