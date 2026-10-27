"""Impact of the 2026-10-25 GBD category fix on Gap and Unified scores. Read-only on docs/data."""
import json, os
from importlib.util import spec_from_file_location, module_from_spec
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
def load(name, path):
    s = spec_from_file_location(name, path); m = module_from_spec(s); s.loader.exec_module(m); return m
sens = load('sens', os.path.join(HERE, '01-r2-sensitivity.py'))
bhem = load('bhem', os.path.join(ROOT, 'PYTHON', '03-03-bhem-compute-metrics.py'))

def main():
    reg = json.load(open(os.path.join(ROOT, 'DATA', 'gbd_disease_registry.json'))); reg = reg.get('diseases', reg)
    integ = json.load(open(os.path.join(sens.DATA, 'integrated.json')))
    meta = {d['id'].replace(' ', '_'): d for d in json.load(open(os.path.join(sens.DATA, 'diseases.json')))['diseases']}
    ct = {}
    for d in json.load(open(os.path.join(sens.DATA, 'clinical_trials.json')))['diseases']:
        ct[d['name'].replace(' ','_').replace('/','_').lower()] = d['intensity']; ct[d['name'].replace(' ','_').lower()] = d['intensity']
    regu = {k.replace(' ', '_'): v for k, v in reg.items()}
    MANUAL = {'Ebola': 20.0, 'Zika_virus': 20.0}  # stored manual overrides, kept
    rows = []
    for d in integ['diseases']:
        m = meta.get(d['disease']); r = regu.get(d['disease']); c = ct.get(d['disease'].lower())
        old_cat = m['category'] if m else None
        new_cat = bhem.map_gbd_category(r['gbd_level2']) if r else old_cat
        g = d['gap_score']
        if m and d['disease'] not in MANUAL:
            # recompute with old category to measure the formula's delta, then apply delta to stored gap
            # (absorbs Leprosy's DALY rounding artefact)
            pubs, dal, gsp = m['research']['globalPublications'], m['burden']['dalysMillions'], m['globalSouthPriority']
            g = min(95.0, d['gap_score'] + sens.gap(pubs, dal, gsp, new_cat) - sens.gap(pubs, dal, gsp, old_cat))
        rows.append(dict(disease=d['disease'], old_cat=old_cat, new_cat=new_cat, gap_old=d['gap_score'], gap=g,
                         sii=d['sii'], ct=c, has_ct=c is not None, injury=d['disease'] in sens.INJ, stored=d['unified_score']))
    df = pd.DataFrame(rows)
    out = []
    ch = df[df.old_cat != df.new_cat]; gch = df[df.gap != df.gap_old]
    out.append(f"Category changed: {len(ch)} diseases; gap score changed: {len(gch)}")
    for r in gch.itertuples(): out.append(f"  {r.disease}: {r.old_cat}->{r.new_cat}  gap {r.gap_old:.0f}->{r.gap:.0f}")
    df['u_new'], w = sens.unified(df, 'gap')
    out.append(f"Re-derived PCA weights: {np.round(w,3)} (published 0.501/0.293/0.206)")
    dis = df[~df.injury & df.u_new.notna()].copy()
    rho = spearmanr(dis.stored, dis.u_new).correlation
    old10 = list(dis.nlargest(10, 'stored').disease); new10 = list(dis.nlargest(10, 'u_new').disease)
    out.append(f"Spearman rho old vs corrected unified (n={len(dis)}): {rho:.4f}; top10 overlap {len(set(old10)&set(new10))}/10")
    out.append(f"  old top10: {old10}\n  new top10: {new10}")
    gsp = {k: v['globalSouthPriority'] for k, v in meta.items()}
    dis['gsp'] = dis.disease.map(gsp)
    out.append(f"GS-priority in corrected top20: {int(dis.nlargest(20,'u_new').gsp.sum())}/20, bottom20: {int(dis.nsmallest(20,'u_new').gsp.sum())}/20")
    ntd = dis[dis.new_cat == 'Neglected']; non = dis[dis.new_cat != 'Neglected']
    out.append(f"Mean unified, Neglected-category vs rest: old {ntd.stored.mean():.1f} vs {non.stored.mean():.1f}; corrected {ntd.u_new.mean():.1f} vs {non.u_new.mean():.1f}")
    df.to_csv(os.path.join(HERE, '05-category-fix-scores.csv'), index=False)
    open(os.path.join(HERE, '06-category-fix-report.txt'), 'w').write('\n'.join(out)); print('\n'.join(out))
if __name__ == '__main__': main()
