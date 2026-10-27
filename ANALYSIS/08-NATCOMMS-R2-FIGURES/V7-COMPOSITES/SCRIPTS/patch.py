from pathlib import Path
R = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/PUBLICATIONS/07-EHR-LINKED-BIOBANKS"
HERE = Path(__file__).parent
s = (R / "PYTHON/06-13-extended-data-fig1-temporal.py").read_text()
reps = [
    ('BASE = Path(__file__).resolve().parents[1]', f'BASE = Path({str(R)!r})'),
    ('OUTPUT = BASE / "ANALYSIS" / "08-NATCOMMS-R2-FIGURES"', f'OUTPUT = Path({str(HERE)!r})'),
    ("f'NTD share: {early_pct:.1f}% (2000-09) to {late_pct:.1f}% (2016-25)\\n{int(growth_factor)}x growth in total pubs'",
     "f'{int(ntd_yearly_c[\"ntd\"].sum())} NTD publications of {int(ntd_yearly_c[\"total\"].sum()):,} (2000-25)\\n{int(growth_factor)}x growth in total pubs'"),
]
for a, b in reps:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
(HERE / "ed1.py").write_text(s)
print("patched")
