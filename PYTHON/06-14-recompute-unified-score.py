"""Refresh docs/data/integrated.json after Gap Scores change (2026-10-25 GBD category fix).

Takes gap_score from docs/data/diseases.json and recomputes the PCA-weighted Unified Score
exactly as 06-10-v7-pca-unified-score.py does, without that script's manuscript edits.
Writes atomically; usage: python PYTHON/06-14-recompute-unified-score.py
"""
import json, os, tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

BASE = Path(__file__).resolve().parents[1]
DOCS = BASE / 'docs' / 'data'


def _key(name):
    """Disease name as used in semantic/integrated files: spaces and slashes become underscores
    ('HIV/AIDS' -> 'HIV_AIDS'). Exact after normalisation; no substring matching."""
    return name.replace(' ', '_').replace('/', '_')


def _ct_lookup(ct_data):
    lk = {}
    for d in ct_data['diseases']:
        lk[d['name'].replace(' ', '_').replace('/', '_')] = d['intensity']
        lk[d['name'].replace(' ', '_')] = d['intensity']
    def get(name):
        if name in lk: return lk[name]
        alt = name.replace('_', ' ').lower()
        for k, v in lk.items():
            if k.replace('_', ' ').lower() == alt: return v
        return None
    return get


def build_rows(diseases_json, semantic, ct_data, exclude=()):
    """One row per disease in semantic.json (the Knowledge corpus defines the disease set),
    with gap_score from diseases.json and ct_equity (trials per million DALYs) from clinical_trials.json."""
    gaps = {_key(d['id']): d['gap']['score'] for d in diseases_json['diseases']}
    get_ct = _ct_lookup(ct_data)
    rows = []
    for name, m in semantic['metrics'].items():
        if name in exclude: continue
        rows.append({'disease': name, 'n_papers': m.get('n_papers'), 'gap_score': gaps.get(name),
                     'ct_equity': get_ct(name), 'sii': m.get('sii'), 'ktp': m.get('ktp'), 'rcc': m.get('rcc'),
                     'mean_drift': (m.get('temporal_drift') or {}).get('mean_drift'),
                     'unified_score': None, 'dimensions_available': 0})
    return rows


def recompute(integrated, diseases_json, ct_data, semantic=None, exclude=()):
    out = deepcopy(integrated)
    if semantic is not None:
        out['diseases'] = build_rows(diseases_json, semantic, ct_data, exclude)
        out['n_diseases'] = len(out['diseases'])
    gaps = {_key(d['id']): d['gap']['score'] for d in diseases_json['diseases']}
    get_ct = _ct_lookup(ct_data)
    rows = out['diseases']
    for d in rows:
        if _key(d['disease']) in gaps: d['gap_score'] = gaps[_key(d['disease'])]
    ct = [get_ct(d['disease']) for d in rows]
    three = [(d, c) for d, c in zip(rows, ct) if d['gap_score'] is not None and d.get('sii') is not None and c is not None]
    X = np.column_stack([[d['gap_score'] for d, _ in three], [1.0/(c+1) for _, c in three], [d['sii'] for d, _ in three]])
    Xn = MinMaxScaler().fit_transform(X)
    pca = PCA(n_components=3).fit(Xn)
    l = pca.components_[0]
    if np.corrcoef(Xn @ l, Xn[:, 0])[0, 1] < 0: l = -l
    w_D, w_T, w_K = np.abs(l) / np.abs(l).sum()
    g = [d['gap_score'] for d in rows if d['gap_score'] is not None]
    s = [d['sii'] for d in rows if d.get('sii') is not None]
    t = [1.0/(c+1) for c in ct if c is not None]
    n = lambda v, lo, hi: (v - lo)/(hi - lo) if hi > lo else 0.5
    for d, c in zip(rows, ct):
        if d['gap_score'] is None or d.get('sii') is None:
            d['unified_score'] = None; d['dimensions_available'] = 0; continue
        dn, kn = n(d['gap_score'], min(g), max(g)), n(d['sii'], min(s), max(s))
        if c is not None:
            d['unified_score'] = float((w_D*dn + w_T*n(1.0/(c+1), min(t), max(t)) + w_K*kn) * 100); d['dimensions_available'] = 3
        else:
            d['unified_score'] = float((w_D*dn + w_K*kn)/(w_D + w_K) * 100); d['dimensions_available'] = 2
    ev = pca.explained_variance_ratio_
    out['weights'].update({'method': f'PCA (PC1 loadings from {len(three)} diseases with all three dimensions)',
                           'discovery': round(float(w_D), 3), 'translation': round(float(w_T), 3),
                           'knowledge': round(float(w_K), 3), 'pc1_explained_variance': round(float(ev[0]), 3),
                           'pc1_pc2_explained_variance': round(float(ev[0] + ev[1]), 3), 'n_complete_cases': len(three)})
    rows.sort(key=lambda d: d['unified_score'] if d['unified_score'] is not None else -1, reverse=True)
    out['n_with_unified_score'] = sum(d['unified_score'] is not None for d in rows)
    vals = [d['unified_score'] for d in rows if d['unified_score'] is not None]
    out['summary'] = {**out.get('summary', {}), 'mean_unified': float(np.mean(vals)), 'sd_unified': float(np.std(vals)),
                      'min_unified': float(min(vals)), 'max_unified': float(max(vals)), 'n_scored': len(vals)}
    return out


EXCLUDE = ('Tracheal',)  # pseudo-disease created by a comma split of the disease list (2026-10-25)
LOW_BURDEN_PERCENTILE = 10  # fixed before inspecting which diseases it flags (2026-10-25)


def add_low_burden_flag(integrated, registry):
    """Flag conditions in the bottom decile of DALYs across all GBD causes. Scores are not changed."""
    out = deepcopy(integrated)
    dalys = {k.replace(' ', '_'): v.get('dalys', 0) for k, v in registry.items() if not k.startswith('_')}
    cut = float(np.percentile(list(dalys.values()), LOW_BURDEN_PERCENTILE))
    for d in out['diseases']:
        v = dalys.get(d['disease'])
        d['dalys'] = v
        d['low_measured_burden'] = bool(v is not None and v < cut)
    out['low_burden_rule'] = {'percentile': LOW_BURDEN_PERCENTILE, 'dalys_threshold': round(cut),
                              'n_causes': len(dalys), 'n_flagged': sum(d['low_measured_burden'] for d in out['diseases'])}
    return out


def main():
    load = lambda f: json.load(open(DOCS / f))
    out = recompute(load('integrated.json'), load('diseases.json'), load('clinical_trials.json'),
                    semantic=load('semantic.json'), exclude=EXCLUDE)
    reg = json.load(open(BASE / 'DATA' / 'gbd_disease_registry.json')); reg = reg.get('diseases', reg)
    out = add_low_burden_flag(out, reg)
    out['generated_at'] = datetime.now().isoformat()
    fd, tmp = tempfile.mkstemp(dir=DOCS, suffix='.json')
    with os.fdopen(fd, 'w') as f: json.dump(out, f, indent=2)
    check = json.load(open(tmp))
    assert len(check['diseases']) == len(out['diseases']) > 0
    os.replace(tmp, DOCS / 'integrated.json')
    print(f"weights D={out['weights']['discovery']} T={out['weights']['translation']} K={out['weights']['knowledge']}; "
          f"n={len(out['diseases'])}")


if __name__ == '__main__':
    main()
