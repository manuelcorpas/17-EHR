"""Translation-dimension numbers quoted in the manuscript, from the mapped trials file,
previous (archived) vs corrected. Positive control: the previous column must reproduce the text."""
import json, os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
WHO = ['Chagas disease', 'Dengue', 'Guinea worm disease', 'Cystic echinococcosis', 'Food-borne trematodiases', 'African trypanosomiasis',
       'Leishmaniasis', 'Leprosy', 'Lymphatic filariasis', 'Onchocerciasis', 'Rabies', 'Scabies', 'Schistosomiasis', 'Cysticercosis', 'Trachoma']
# The manuscript's 'NTD-mapped trials' (3,703) reproduces only with the WHO list plus yellow fever, Ebola
# and the GBD residual 'Other neglected tropical diseases' (18 causes); recorded here as the paper's operational set.
NTD18 = WHO + ['Yellow fever', 'Ebola', 'Other neglected tropical diseases']
COLS = ['nct_id', 'phase', 'study_type', 'start_year', 'cause_count', 'gbd_causes_str', 'gbd_level2_str', 'global_south_priority']

def numbers(studies_csv, ct_json):
    s = pd.read_csv(studies_csv, usecols=COLS); c = {d['name']: d for d in json.load(open(ct_json))['diseases']}
    causes = s.gbd_causes_str.fillna('').str.split('|'); lvl2 = s.gbd_level2_str.fillna('')
    is_who = causes.apply(lambda L: any(x in WHO for x in L)); is_ntdm = lvl2.str.contains('NTDs and Malaria')
    is_18 = causes.apply(lambda L: any(x in NTD18 for x in L))
    obs = s.study_type.astype(str).str.upper().eq('OBSERVATIONAL')
    # Phase definitions that reproduce the submitted text (5.2/8.3 and 13.2/13.3 within rounding): exact PHASE1,
    # exact PHASE3 or PHASE4, denominators all trials incl. those with no phase (observational)
    ph = s.phase.fillna('NA').astype(str).str.upper(); p1 = ph.eq('PHASE1'); late = ph.isin(['PHASE3', 'PHASE4'])
    out = {'total_trials': len(s), 'mapped_trials': int((s.cause_count > 0).sum()),
           'ntd_trials_WHO_list': int(is_who.sum()), 'ntd_trials_level2_NTDs_and_malaria': int(is_ntdm.sum()),
           'ntd_trials_18_set_as_in_text': int(is_18.sum()), 'ntd_pct_18': round(100 * is_18.mean(), 2),
           'obs_pct_ntd_18': round(100 * obs[is_18].mean(), 1), 'obs_pct_other_18': round(100 * obs[~is_18].mean(), 1),
           'phase1_pct_ntd_18': round(100 * p1[is_18].mean(), 1), 'phase1_pct_other_18': round(100 * p1[~is_18].mean(), 1),
           'late_pct_ntd_18': round(100 * late[is_18].mean(), 1), 'late_pct_other_18': round(100 * late[~is_18].mean(), 1),
           'ntd_pct_WHO': round(100 * is_who.mean(), 2), 'ntd_pct_level2': round(100 * is_ntdm.mean(), 2),
           'obs_pct_ntd_WHO': round(100 * obs[is_who].mean(), 1), 'obs_pct_ntd_level2': round(100 * obs[is_ntdm].mean(), 1),
           'obs_pct_other': round(100 * obs[~is_who].mean(), 1), 'obs_pct_other_mapped': round(100 * obs[(~is_who) & (s.cause_count > 0)].mean(), 1)}
    g = lambda n, k: c[n][k] if n in c else None
    out.update({'lf_trials': g('Lymphatic filariasis', 'trials'), 'diabetes_trials': g('Diabetes mellitus', 'trials'),
                'ratio_diabetes_lf': round(g('Diabetes mellitus', 'trials') / g('Lymphatic filariasis', 'trials'), 0),
                'schisto_trials': g('Schistosomiasis', 'trials'), 'cysticercosis_trials': g('Cysticercosis', 'trials'),
                'ihd_trials': g('Ischemic heart disease', 'trials'), 'ihd_intensity': round(g('Ischemic heart disease', 'intensity'), 0),
                'lf_intensity': round(g('Lymphatic filariasis', 'intensity'), 0), 'oncho_trials': g('Onchocerciasis', 'trials')})
    temporal = {t['year']: t for t in json.load(open(ct_json))['temporal']}
    out.update({'gs_share_2000': round(temporal[2000]['gsPct'], 1), 'gs_share_2020': round(temporal[2020]['gsPct'], 1),
                'gs_share_2024': round(temporal[2024]['gsPct'], 1), 'gs_share_2025': round(temporal[2025]['gsPct'], 1)})
    out.update({k: v for k, v in json.load(open(ct_json))['globalSouthAnalysis'].items() if k in ('gsTrialsPct', 'gsIntensity', 'nonGsIntensity', 'intensityGap')})
    return out

def main():
    A = os.path.join(ROOT, 'ARCHIVE', '2026-10-25-PRE-MESH-FIX')
    prev = numbers(os.path.join(A, 'DATA', 'heim_ct_studies_mapped.csv'), os.path.join(A, 'docs-data', 'clinical_trials.json'))
    corr = numbers(os.path.join(ROOT, 'DATA', 'heim_ct_studies_mapped.csv'), os.path.join(ROOT, 'docs', 'data', 'clinical_trials.json'))
    t = pd.DataFrame({'previous': prev, 'corrected': corr}); t['changed'] = t.previous.astype(str) != t.corrected.astype(str)
    t.to_csv(os.path.join(HERE, '14-translation-numbers.csv')); pd.set_option('display.width', 200); print(t.to_string())
if __name__ == '__main__': main()
