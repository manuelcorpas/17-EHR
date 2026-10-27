"""Weight perturbation on the corrected data (replaces the lost ED Fig 2 renderer's analysis).
17 schemes per dimension, varying one weight and keeping the other two in their PCA ratio."""
import json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
def main():
    I = json.load(open(os.path.join(ROOT, 'docs', 'data', 'integrated.json')))
    from importlib.util import spec_from_file_location, module_from_spec
    sp = spec_from_file_location('ru', os.path.join(ROOT, 'PYTHON', '06-14-recompute-unified-score.py'))
    ru = module_from_spec(sp); sp.loader.exec_module(ru)
    get_ct = ru._ct_lookup(json.load(open(os.path.join(ROOT, 'docs', 'data', 'clinical_trials.json'))))
    df = pd.DataFrame(I['diseases']); df['ct'] = [get_ct(x) for x in df.disease]
    n_all = lambda s: s  # normalisation over all 175 rows, as in 06-14
    df = df.copy()
    n = lambda s: (s - s.min()) / (s.max() - s.min())
    D, K = n(df.gap_score), n(df.sii); t = 1 / (df.ct + 1); T = (t - t.min()) / (t.max() - t.min())
    keep = df.unified_score.notna().values
    w0 = np.array([I['weights'][k] for k in ('discovery', 'translation', 'knowledge')])
    def score(w):
        three = 100 * (w[0] * D + w[1] * T + w[2] * K); two = 100 * (w[0] * D + w[2] * K) / (w[0] + w[2])
        return np.where(df.ct.notna(), three, two)[keep]
    base = score(w0); diff = np.abs(base - df.unified_score.values[keep]).max()
    print('max abs diff vs stored (weights stored to 3 dp):', round(diff, 4))
    assert diff < 0.1 and spearmanr(base, df.unified_score.values[keep]).correlation > 0.9999, 'does not reproduce stored scores'
    out = {}
    for i, (lo, hi) in enumerate([(0.30, 0.70), (0.10, 0.50), (0.05, 0.40)]):
        rhos = []
        for v in np.linspace(lo, hi, 17):
            w = w0.copy(); others = [j for j in range(3) if j != i]
            w[others] = w0[others] / w0[others].sum() * (1 - v); w[i] = v
            rhos.append(spearmanr(base, score(w)).correlation)
        out[['discovery', 'translation', 'knowledge'][i]] = (round(min(rhos), 3), round(max(rhos), 3))
    out['min_all'] = min(v[0] for v in out.values())
    out['equal_weights'] = round(spearmanr(base, score(np.array([1/3, 1/3, 1/3]))).correlation, 3)
    rng = np.random.default_rng(20261025)
    dr = [spearmanr(base, score(rng.dirichlet([1, 1, 1]))).correlation for _ in range(1000)]
    out['dirichlet_1000'] = (round(min(dr), 3), round(float(np.median(dr)), 3))
    json.dump(out, open(os.path.join(HERE, '12-weight-perturbation.json'), 'w'), indent=2); print(out)
if __name__ == '__main__': main()
