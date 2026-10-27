"""Figure annotations must be computed from data, never typed in."""
import json
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location('figs', ROOT / 'PYTHON' / '06-heim-publication-figures.py')
figs = module_from_spec(spec); spec.loader.exec_module(figs)

def test_paths_are_relative_to_repo():
    assert figs.BASE_PATH == ROOT and 'superintelligent' not in str(figs.BASE_PATH)

def test_weights_label_reads_integrated_weights():
    w = {'weights': {'discovery': 0.526, 'translation': 0.229, 'knowledge': 0.246}}
    assert figs.weights_label(w) == 'Weights: D=0.53, T=0.23, K=0.25 (PCA)'

def test_eas_summary_counts_and_percentage():
    df = pd.DataFrame({'name': ['UK Biobank', 'a', 'b', 'c'], 'eas': [84.0, 50, 10, 5],
                       'category': ['High', 'Moderate', 'Low', 'Low']})
    s = figs.eas_summary(df)
    assert s['counts'] == {'High': 1, 'Moderate': 1, 'Low': 2} and s['pct_low'] == 50 and s['uk_eas'] == 84.0

def test_palette_has_no_red_green_pair():
    banned = {'#27AE60', '#E74C3C'}
    src = (ROOT / 'PYTHON' / '06-heim-publication-figures.py').read_text()
    assert not any(c in src for c in banned)

def test_global_south_comes_from_diseases_json():
    gs = figs.global_south_set(json.load(open(ROOT / 'docs' / 'data' / 'diseases.json')))
    assert 'diphtheria' in gs and 'otitis media' in gs and 'urticaria' not in gs

def test_ed1_trend_wording_is_computed():
    spec2 = spec_from_file_location('ed1', ROOT / 'PYTHON' / '06-13-extended-data-fig1-temporal.py')
    src = (ROOT / 'PYTHON' / '06-13-extended-data-fig1-temporal.py').read_text()
    assert '56%' not in src and "set_title('Clinical trial equity is declining'" not in src  # no typed-in verdict at the plot call
    assert 'def trend_wording' in src and 'ax.set_title(title_b' in src
    ns = {}; exec(src[src.index('def trend_wording'):src.index('\n\n', src.index('def trend_wording'))], ns)
    assert ns['trend_wording'](-0.6, 1e-5, 55.5, 32.3)[0].startswith('Clinical trial equity is declining')
    assert ns['trend_wording'](0.027, 0.65, 11.5, 11.1)[0].startswith('Global South share of trials is flat')
