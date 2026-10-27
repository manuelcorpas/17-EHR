"""NCOMMS-26-040266-T round 2: R4.1 complete-case, R4.2 no +10 GS penalty, R3.5 specificity.
Reads frozen docs/data/*.json, writes only into this folder."""
import json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'docs', 'data')
INFECTIOUS = {"Infectious", "Neglected"}; NEGLECTED = {"Neglected", "Maternal/Child", "Nutritional"}
INJ = {'Conflict_and_terrorism','Animal_contact','Drowning','Road_injuries','Self-harm','Interpersonal_violence','Falls',
 'Fire,_heat,_and_hot_substances','Poisonings','Exposure_to_mechanical_forces','Foreign_body','Environmental_heat_and_cold_exposure',
 'Other_transport_injuries','Executions_and_police_conflict','Other_unintentional_injuries','Adverse_effects_of_medical_treatment',
 'Paralytic_ileus_and_intestinal_obstruction'}

def gap(pubs, dalys, gsp, cat, penalty=True):  # copy of 03-03 compute_research_gap_score
    if pubs == 0: return 95.0
    g = None
    if cat in INFECTIOUS: g = 90. if pubs<10 else 80. if pubs<25 else 70. if pubs<50 else 60. if pubs<100 else None
    elif cat in NEGLECTED: g = 92. if pubs<10 else 82. if pubs<25 else 72. if pubs<50 else 65. if pubs<100 else None
    if g is None:
        r = pubs/dalys if dalys and dalys > 0 else pubs
        g = next((v for t, v in [(100,10.),(50,20.),(25,30.),(10,40.),(5,50.),(2,60.),(1,70.),(.5,80.)] if r >= t), 85.)
    if penalty and gsp and pubs < 50: g = min(95.0, g + 10.0)
    return g

def unified(df, gcol, weights=None):
    three = df[df.gap.notna() & df.sii.notna() & df.has_ct]
    X = np.column_stack([three[gcol], 1/(three.ct+1), three.sii])
    if weights is None:
        Xn = MinMaxScaler().fit_transform(X); p = PCA(3).fit(Xn); l = p.components_[0]
        if np.corrcoef(Xn @ l, Xn[:,0])[0,1] < 0: l = -l
        weights = np.abs(l)/np.abs(l).sum()
    wD, wT, wK = weights
    n = lambda v, s: (v - s.min())/(s.max() - s.min())
    allg, alls = df[gcol].dropna(), df.sii.dropna(); allt = 1/(df.ct.dropna()+1)
    out = []
    for _, r in df.iterrows():
        if pd.isna(r[gcol]) or pd.isna(r.sii): out.append(np.nan); continue
        dn, kn = n(r[gcol], allg), n(r.sii, alls)
        out.append(100*(wD*dn + wT*n(1/(r.ct+1), allt) + wK*kn) if r.has_ct else 100*(wD*dn + wK*kn)/(wD+wK))
    return np.array(out), weights

def main():
    integ = json.load(open(os.path.join(DATA, 'integrated.json')))
    meta = {d['id'].replace(' ', '_'): d for d in json.load(open(os.path.join(DATA, 'diseases.json')))['diseases']}
    ct = {}
    for d in json.load(open(os.path.join(DATA, 'clinical_trials.json')))['diseases']:
        ct[d['name'].replace(' ','_').replace('/','_').lower()] = d['intensity']; ct[d['name'].replace(' ','_').lower()] = d['intensity']
    reg = json.load(open(os.path.join(DATA, '..', '..', 'DATA', 'gbd_disease_registry.json'))); reg = reg.get('diseases', reg)
    exact_dalys = {k.replace(' ', '_'): v.get('dalys', 0) / 1e6 for k, v in reg.items()}  # unrounded, as the pipeline uses
    rows = []
    for d in integ['diseases']:
        m = meta.get(d['disease']); c = ct.get(d['disease'].lower())
        rows.append(dict(disease=d['disease'], gap=d['gap_score'], sii=d['sii'], ct=c, has_ct=c is not None,
            stored=d['unified_score'], injury=d['disease'] in INJ, matched=m is not None,
            gsp=m['globalSouthPriority'] if m else None, category=m['category'] if m else None,
            pubs=m['research']['globalPublications'] if m else None,
            dalys=exact_dalys.get(d['disease'], m['burden']['dalysMillions'] if m else None)))
    df = pd.DataFrame(rows)
    log = []
    # positive control 1: gap function reproduces stored gap
    df['gap_recalc'] = [gap(r.pubs, r.dalys, r.gsp, r.category) if r.matched else np.nan for r in df.itertuples()]
    mm = df[df.matched & (df.gap_recalc != df.gap)]
    log.append(f"CONTROL gap: {df.matched.sum()}/{len(df)} matched, {len(mm)} mismatches vs stored gap: {mm[['disease','gap','gap_recalc','pubs','category','gsp']].to_dict('records')}")
    # positive control 2: unified reproduces stored
    df['u_base'], w = unified(df, 'gap')
    diff = (df.u_base - df.stored).abs().max()
    log.append(f"CONTROL unified: weights {np.round(w,3)}, max abs diff vs stored {diff:.2e}")
    # Known non-formula stored gaps: Ebola, Zika = manual overrides (DALYs 0, adjustedNote set; no penalty applied);
    # Leprosy = DALYs rounded to 0.02 in diseases.json, stored 40 = 30 + 10 penalty.
    OVERRIDE_NOPEN = {}  # pre-fix values were {'Ebola': 20, 'Zika_virus': 20, 'Leprosy': 30}; none needed after 2026-10-25 category fix
    unexplained = set(mm.disease) - set(OVERRIDE_NOPEN)
    log.append(f"CONTROL unexplained gap mismatches: {sorted(unexplained)}; unmatched: {sorted(df[~df.matched].disease)}")
    if diff > 1e-6 or unexplained: log.append("CONTROL FAILED: results below are INCOMPLETE")
    dis = df[~df.injury & df.u_base.notna()].copy()
    dis['rank_base'] = dis.u_base.rank(ascending=False, method='min')
    top10 = set(dis.nsmallest(10, 'rank_base').disease)
    def compare(name, score_col, sub):
        s = sub.dropna(subset=[score_col]).copy(); s['r'] = s[score_col].rank(ascending=False, method='min')
        rho = spearmanr(s.u_base, s[score_col]).correlation
        base_top = set(s.nsmallest(10, 'rank_base' if 'rank_base_sub' not in s else 'rank_base_sub').disease)
        new_top = set(s.nsmallest(10, 'r').disease)
        log.append(f"\n## {name}\nn={len(s)}  Spearman rho vs baseline={rho:.4f}  top10 overlap={len(base_top & new_top)}/10")
        log.append(f"  dropped from top10: {sorted(base_top-new_top)}\n  entered top10: {sorted(new_top-base_top)}")
        return s
    # A1 complete case: 3-dim diseases only, same weights; and re-derived PCA identical (PCA already used 86)
    cc = dis[dis.has_ct].copy(); cc['rank_base_sub'] = cc.u_base.rank(ascending=False, method='min')
    log.append(f"\nA1: {len(cc)} complete-case non-injury diseases ({df.has_ct.sum()} incl. injuries with all dims); "
               f"baseline top10 members with 3 dims: {sum(dis.set_index('disease').loc[list(top10)].has_ct)}/10")
    s = cc.sort_values('u_base', ascending=False)
    log.append("A1 complete-case top 10 (3-dim scores unchanged; ranking among complete cases):")
    for i, r in enumerate(s.head(10).itertuples(), 1): log.append(f"  {i}. {r.disease} {r.u_base:.1f} (full-rank {int(r.rank_base)})")
    # A1b: is the 2-dim fallback inflating? compare score distributions
    log.append(f"A1b mean unified: 2-dim {dis[~dis.has_ct].u_base.mean():.1f} (n={(~dis.has_ct).sum()}), 3-dim {cc.u_base.mean():.1f}")
    # A1c: score 3-dim diseases with D+K only, rho vs their 3-dim score
    cc['u_dk'] = [100*(w[0]*(r.gap-df.gap.min())/(df.gap.max()-df.gap.min()) + w[2]*(r.sii-df.sii.min())/(df.sii.max()-df.sii.min()))/(w[0]+w[2]) for r in cc.itertuples()]
    log.append(f"A1c on the 86-type set, D+K fallback vs full 3-dim score: rho={spearmanr(cc.u_base, cc.u_dk).correlation:.4f}, "
               f"top10 overlap={len(set(cc.nlargest(10,'u_base').disease)&set(cc.nlargest(10,'u_dk').disease))}/10")
    # A2 no penalty
    df['gap_nopen'] = [gap(r.pubs, r.dalys, r.gsp, r.category, penalty=False) if r.matched and r.disease not in OVERRIDE_NOPEN else OVERRIDE_NOPEN.get(r.disease, r.gap) for r in df.itertuples()]
    changed = df[df.gap_nopen != df.gap]
    log.append(f"\nA2: penalty changes the Gap Score of {len(changed)} diseases")
    df['u_nopen_fixedw'], _ = unified(df, 'gap_nopen', weights=w)
    df['u_nopen_repca'], w2 = unified(df, 'gap_nopen')
    log.append(f"A2 re-derived PCA weights without penalty: {np.round(w2,3)}")
    dis = dis.join(df.set_index('disease')[['u_nopen_fixedw','u_nopen_repca','gap_nopen']], on='disease')
    compare("A2 no penalty, baseline weights", 'u_nopen_fixedw', dis)
    compare("A2 no penalty, re-derived PCA weights", 'u_nopen_repca', dis)
    # A4 specificity: where do well-studied high-income conditions land
    dis['pct'] = 100*dis.rank_base/len(dis)
    probes = [x for x in dis.disease if any(k in x.lower() for k in ['breast','ischemic_heart','ischaemic','diabetes','alzheimer','stroke','prostate','colon','lung','hiv','asthma','copd','depress'])]
    log.append(f"\nA4 rank of well-studied conditions (of {len(dis)}; 1 = most neglected):")
    for r in dis[dis.disease.isin(probes)].sort_values('rank_base').itertuples():
        log.append(f"  {r.disease}: rank {int(r.rank_base)} (score {r.u_base:.1f}, dims={'3' if r.has_ct else '2'})")
    gs = dis.join(df.set_index('disease')['gsp'], on='disease', rsuffix='_')
    log.append(f"A4 GS-priority among top 20: {int(gs.nsmallest(20,'rank_base').gsp.sum())}/20; among bottom 20: {int(gs.nlargest(20,'rank_base').gsp.sum())}/20")
    dis.to_csv(os.path.join(HERE, '02-r2-sensitivity-scores.csv'), index=False)
    open(os.path.join(HERE, '03-r2-sensitivity-report.txt'), 'w').write('\n'.join(log)); print('\n'.join(log))

if __name__ == '__main__': main()
