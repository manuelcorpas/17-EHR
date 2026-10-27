"""ED Fig 2 statistics must agree with the sensitivity report they illustrate (no typed-in numbers)."""
import re
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location('ed2', ROOT / 'PYTHON' / '08-01-ed-fig2-sensitivity.py')
ed2 = module_from_spec(spec); spec.loader.exec_module(ed2)

def test_stats_match_sensitivity_report():
    rep = (ROOT / 'ANALYSIS' / '03-11-NATCOMMS-R2-SENSITIVITY' / '03-r2-sensitivity-report.txt').read_text()
    s = ed2.compute()['stats']
    m = re.search(r'A2 no penalty, baseline weights\nn=\d+\s+Spearman rho vs baseline=([0-9.]+)\s+top10 overlap=(\d+)/10', rep)
    assert m and abs(s['rho_nopen'] - float(m.group(1))) < 5e-4 and s['top10_nopen'] == int(m.group(2))
    m = re.search(r'A1c .*?rho=([0-9.]+), top10 overlap=(\d+)/10', rep)
    assert m and abs(s['rho_fallback'] - float(m.group(1))) < 5e-4 and s['top10_fallback'] == int(m.group(2))
    assert 0 < s['rho_correction'] <= 1 and 0 <= s['top10_correction'] <= 10
