import json
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; ARC = ROOT / 'ARCHIVE' / '2026-10-25-PRE-MESH-FIX'
spec = spec_from_file_location('g', ROOT / 'PYTHON' / '04-05-ct-generate-docs-json.py'); g = module_from_spec(spec); spec.loader.exec_module(g)

def test_reproduces_archived_clinical_trials_json():
    studies = pd.read_csv(ARC / 'DATA' / 'heim_ct_studies_mapped.csv', usecols=['nct_id', 'start_year', 'cause_count', 'global_south_priority'])
    reg = json.load(open(ARC / 'DATA' / 'heim_ct_disease_registry.json')); ref = json.load(open(ARC / 'docs-data' / 'clinical_trials.json'))
    out = g.recompute(studies, reg, ref)
    a = {d['name']: d for d in out['diseases']}; b = {d['name']: d for d in ref['diseases']}
    assert set(a) == set(b)
    for k in b:
        assert a[k]['trials'] == b[k]['trials'] and abs(a[k]['intensity'] - b[k]['intensity']) < 1e-6 and a[k]['category'] == b[k]['category'], k
    for k in ('totalTrials', 'totalDiseaseTrialMappings', 'totalDALYs', 'diseaseCategories'):
        assert out['summary'][k] == ref['summary'][k], (k, out['summary'][k], ref['summary'][k])
    assert out['globalSouthAnalysis'] == ref['globalSouthAnalysis'], (out['globalSouthAnalysis'], ref['globalSouthAnalysis'])
    ca = {c['name']: c for c in out['categories']}; cb = {c['name']: c for c in ref['categories']}
    assert set(ca) == set(cb) and all(ca[k]['trials'] == cb[k]['trials'] and ca[k]['diseaseCount'] == cb[k]['diseaseCount'] for k in cb)
    ta = {t['year']: t for t in out['temporal']}; tb = {t['year']: t for t in ref['temporal']}
    assert set(ta) == set(tb) and all(ta[y]['trials'] == tb[y]['trials'] and ta[y]['gsTrials'] == tb[y]['gsTrials'] for y in tb), [(y, ta[y], tb[y]) for y in tb if ta[y]['trials'] != tb[y]['trials']][:3]
