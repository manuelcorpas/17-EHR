import json
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location('ru', ROOT / 'PYTHON' / '06-14-recompute-unified-score.py')
ru = module_from_spec(spec); spec.loader.exec_module(ru)
ARC = ROOT / 'ARCHIVE' / '2026-10-25-PRE-CATEGORY-FIX' / 'docs-data'

def test_reproduces_published_scores_on_archived_inputs():
    integ = json.load(open(ARC / 'integrated.json'))
    out = ru.recompute(integ, json.load(open(ARC / 'diseases.json')), json.load(open(ARC / 'clinical_trials.json')))
    old = {d['disease']: d['unified_score'] for d in integ['diseases']}
    for d in out['diseases']:
        if old[d['disease']] is None: assert d['unified_score'] is None
        else: assert abs(d['unified_score'] - old[d['disease']]) < 1e-9, d['disease']
    assert abs(out['weights']['discovery'] - 0.501) < 1e-3

def test_gap_scores_taken_from_diseases_json():
    integ = json.load(open(ARC / 'integrated.json')); dj = json.load(open(ARC / 'diseases.json'))
    for d in dj['diseases']:
        if d['id'] == 'Trachoma': d['gap']['score'] = 90.0
    out = ru.recompute(integ, dj, json.load(open(ARC / 'clinical_trials.json')))
    assert next(d for d in out['diseases'] if d['disease'] == 'Trachoma')['gap_score'] == 90.0

def test_low_burden_flag_is_bottom_decile_and_does_not_change_scores():
    integ = json.load(open(ARC / 'integrated.json'))
    reg = json.load(open(ROOT / 'DATA' / 'gbd_disease_registry.json')); reg = reg.get('diseases', reg)
    base = ru.recompute(integ, json.load(open(ARC / 'diseases.json')), json.load(open(ARC / 'clinical_trials.json')))
    out = ru.add_low_burden_flag(base, reg)
    flag = {d['disease']: d['low_measured_burden'] for d in out['diseases']}
    assert flag['Zika_virus'] and flag['Ebola'] and flag['Guinea_worm_disease']
    assert not flag['Malaria'] and not flag['Breast_cancer']
    assert sum(flag.values()) == 18
    assert out['low_burden_rule']['percentile'] == 10
    assert [d['unified_score'] for d in out['diseases']] == [d['unified_score'] for d in base['diseases']]

def test_build_from_sources_reproduces_archived_integrated():
    ARC2 = ROOT / 'ARCHIVE' / '2026-10-25-PRE-MESH-FIX' / 'docs-data'
    L = lambda f: json.load(open(ARC2 / f))
    ref = L('integrated.json')
    dj = L('diseases.json')
    # The archived file carried a pseudo-disease 'Tracheal' with gap_score 30 (a name-split artefact
    # scored through a substring fallback); inject it so the archived file is reproducible exactly.
    dj['diseases'].append({'id': 'Tracheal', 'gap': {'score': 30.0}})
    out = ru.recompute(ref, dj, L('clinical_trials.json'), semantic=L('semantic.json'))
    a = {d['disease']: d for d in out['diseases']}; b = {d['disease']: d for d in ref['diseases']}
    assert set(a) == set(b) and len(a) == 175
    for k in b:
        for f in ('gap_score', 'ct_equity', 'sii', 'ktp', 'rcc', 'mean_drift', 'n_papers', 'unified_score'):
            x, y = a[k][f], b[k][f]
            assert (x is None and y is None) or abs(x - y) < 1e-9, (k, f, x, y)
        assert a[k]['dimensions_available'] == b[k]['dimensions_available']
    assert 'Tracheal' in a
    out2 = ru.recompute(ref, dj, L('clinical_trials.json'), semantic=L('semantic.json'), exclude=('Tracheal',))
    assert 'Tracheal' not in {d['disease'] for d in out2['diseases']} and out2['n_diseases'] == 174

def test_summary_block_is_recomputed():
    ARC2 = ROOT / 'ARCHIVE' / '2026-10-25-PRE-MESH-FIX' / 'docs-data'
    L = lambda f: json.load(open(ARC2 / f))
    out = ru.recompute(L('integrated.json'), L('diseases.json'), L('clinical_trials.json'), semantic=L('semantic.json'), exclude=('Tracheal',))
    vals = [d['unified_score'] for d in out['diseases'] if d['unified_score'] is not None]
    import statistics
    assert abs(out['summary']['mean_unified'] - statistics.mean(vals)) < 1e-9
