#!/usr/bin/env python3
"""
HEIM SEMANTIC ANALYSIS - DATA QUALITY AUDIT
============================================

Audits the PubMed data collection for completeness and accuracy.
Flags diseases where paper counts seem incorrect based on:
1. Expected counts from disease prominence
2. Trial counts as sanity check
3. Known high-volume diseases

OUTPUTS:
- DATA/05-SEMANTIC/AUDIT/audit_report.json
- DATA/05-SEMANTIC/AUDIT/audit_summary.csv
- DATA/05-SEMANTIC/AUDIT/flagged_diseases.csv

USAGE:
    python 05-05-heim-sem-audit.py
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "DATA"
SEMANTIC_DIR = DATA_DIR / "05-SEMANTIC"
AUDIT_DIR = SEMANTIC_DIR / "AUDIT"
DOCS_DATA_DIR = BASE_DIR / "docs" / "data"

# Expected paper counts for major diseases (approximate, from PubMed)
# These are rough estimates based on disease prominence
EXPECTED_PAPERS = {
    # Cancers - very high volume
    "Breast cancer": 250000,
    "Lung cancer": 200000,
    "Tracheal, bronchus, and lung cancer": 200000,
    "Prostate cancer": 150000,
    "Colon and rectum cancer": 100000,
    "Liver cancer": 80000,
    "Pancreatic cancer": 60000,
    "Brain and central nervous system cancer": 80000,
    "Stomach cancer": 60000,
    "Ovarian cancer": 50000,
    "Leukemia": 150000,

    # Cardiovascular - very high volume
    "Ischemic heart disease": 150000,
    "Stroke": 200000,
    "Hypertensive heart disease": 80000,
    "Heart failure": 100000,
    "Atrial fibrillation and flutter": 50000,

    # Metabolic/Endocrine
    "Diabetes mellitus": 200000,
    "Obesity": 100000,

    # Neurological
    "Alzheimer's disease and other dementias": 150000,
    "Parkinson's disease": 80000,
    "Epilepsy": 50000,
    "Multiple sclerosis": 50000,
    "Schizophrenia": 80000,
    "Depressive disorders": 100000,

    # Infectious
    "HIV/AIDS": 150000,
    "Tuberculosis": 100000,
    "Malaria": 60000,
    "COVID-19": 200000,
    "Hepatitis": 80000,

    # Respiratory
    "Chronic obstructive pulmonary disease": 60000,
    "Asthma": 100000,
    "Pneumonia": 80000,
    "Lower respiratory infections": 50000,

    # Other high-volume
    "Rheumatoid arthritis": 80000,
    "Osteoarthritis": 60000,
    "Chronic kidney disease": 50000,
    "Inflammatory bowel disease": 50000,
    "Cirrhosis and other chronic liver diseases": 40000,
}

# Minimum expected papers based on trial counts (papers should be >= trials * 0.5 for major diseases)
MIN_PAPER_TO_TRIAL_RATIO = 0.1  # Very conservative - at least 10% of trial count


def load_data():
    """Load integrated metrics and clinical trials data."""
    integrated_file = DOCS_DATA_DIR / "integrated.json"
    ct_file = DOCS_DATA_DIR / "clinical_trials.json"

    with open(integrated_file) as f:
        integrated = json.load(f)

    with open(ct_file) as f:
        ct = json.load(f)

    return integrated, ct


def audit_diseases(integrated, ct):
    """Audit all diseases for data quality issues."""

    ct_diseases = {d['name']: d for d in ct.get('diseases', [])}

    audit_results = []

    for d in integrated.get('diseases', []):
        disease_name = d['disease'].replace('_', ' ')
        n_papers = d.get('n_papers', 0)
        sii = d.get('sii')

        # Get trial data
        ct_d = ct_diseases.get(disease_name, {})
        n_trials = ct_d.get('trials', 0)
        dalys = ct_d.get('dalys', 0)

        # Determine expected paper count
        expected = EXPECTED_PAPERS.get(disease_name, None)

        # Calculate expected minimum from trials
        expected_from_trials = n_trials * MIN_PAPER_TO_TRIAL_RATIO if n_trials > 1000 else None

        # Flags
        flags = []
        severity = "OK"

        # Check 1: Very low paper count
        if n_papers < 50:
            flags.append("VERY_LOW_PAPERS")
            severity = "CRITICAL"
        elif n_papers < 500:
            flags.append("LOW_PAPERS")
            if severity != "CRITICAL":
                severity = "WARNING"

        # Check 2: Paper count vs expected
        if expected and n_papers < expected * 0.1:
            flags.append("FAR_BELOW_EXPECTED")
            severity = "CRITICAL"
        elif expected and n_papers < expected * 0.3:
            flags.append("BELOW_EXPECTED")
            if severity != "CRITICAL":
                severity = "WARNING"

        # Check 3: Paper count vs trial count
        if expected_from_trials and n_papers < expected_from_trials:
            flags.append("LOW_VS_TRIALS")
            if severity == "OK":
                severity = "WARNING"

        # Check 4: High trials but very low papers (major red flag)
        if n_trials > 10000 and n_papers < 1000:
            flags.append("HIGH_TRIALS_LOW_PAPERS")
            severity = "CRITICAL"

        # Check 5: High DALYs but low papers
        if dalys > 20 and n_papers < 5000:
            flags.append("HIGH_BURDEN_LOW_PAPERS")
            if severity == "OK":
                severity = "WARNING"

        audit_results.append({
            'disease': disease_name,
            'n_papers': n_papers,
            'n_trials': n_trials,
            'dalys': dalys,
            'expected_papers': expected,
            'sii': sii,
            'flags': flags,
            'severity': severity,
            'flag_count': len(flags),
            'needs_review': len(flags) > 0
        })

    return audit_results


def generate_report(audit_results):
    """Generate audit report with summary statistics."""

    df = pd.DataFrame(audit_results)

    # Summary stats
    total = len(df)
    critical = len(df[df['severity'] == 'CRITICAL'])
    warning = len(df[df['severity'] == 'WARNING'])
    ok = len(df[df['severity'] == 'OK'])

    # Specific flag counts
    flag_counts = {}
    for _, row in df.iterrows():
        for flag in row['flags']:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    report = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'total_diseases': total,
            'critical': critical,
            'warning': warning,
            'ok': ok,
            'critical_pct': round(100 * critical / total, 1),
            'warning_pct': round(100 * warning / total, 1),
            'ok_pct': round(100 * ok / total, 1)
        },
        'flag_counts': flag_counts,
        'critical_diseases': df[df['severity'] == 'CRITICAL'][['disease', 'n_papers', 'n_trials', 'flags']].to_dict('records'),
        'recommendations': [
            "CRITICAL diseases need re-collection with corrected search queries",
            "WARNING diseases should be reviewed for search term accuracy",
            "Use MeSH terms and synonyms instead of exact GBD names",
            "Validate against PubMed web interface for spot checks"
        ]
    }

    return report, df


def main():
    print("\n" + "="*70)
    print(" HEIM SEMANTIC ANALYSIS - DATA QUALITY AUDIT")
    print("="*70)

    # Create output directory
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    integrated, ct = load_data()

    # Run audit
    print("Auditing diseases...")
    audit_results = audit_diseases(integrated, ct)

    # Generate report
    print("Generating report...")
    report, df = generate_report(audit_results)

    # Save outputs
    report_file = AUDIT_DIR / "audit_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_file}")

    summary_file = AUDIT_DIR / "audit_summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"  Saved: {summary_file}")

    flagged_file = AUDIT_DIR / "flagged_diseases.csv"
    flagged = df[df['needs_review']].sort_values('severity', ascending=False)
    flagged.to_csv(flagged_file, index=False)
    print(f"  Saved: {flagged_file}")

    # Print summary
    print("\n" + "="*70)
    print(" AUDIT SUMMARY")
    print("="*70)
    print(f"  Total diseases: {report['summary']['total_diseases']}")
    print(f"  CRITICAL: {report['summary']['critical']} ({report['summary']['critical_pct']}%)")
    print(f"  WARNING:  {report['summary']['warning']} ({report['summary']['warning_pct']}%)")
    print(f"  OK:       {report['summary']['ok']} ({report['summary']['ok_pct']}%)")

    print("\n  Flag breakdown:")
    for flag, count in sorted(report['flag_counts'].items(), key=lambda x: -x[1]):
        print(f"    {flag}: {count}")

    print("\n  CRITICAL diseases requiring re-collection:")
    for d in report['critical_diseases'][:15]:
        print(f"    - {d['disease']}: {d['n_papers']} papers, {d['n_trials']} trials")

    if len(report['critical_diseases']) > 15:
        print(f"    ... and {len(report['critical_diseases']) - 15} more")

    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print("  1. Review flagged_diseases.csv")
    print("  2. Update GBD-to-MeSH mapping (gbd_mesh_mapping.json)")
    print("  3. Re-run 05-01-heim-sem-fetch.py with corrected queries")
    print("  4. Re-run embedding and metric computation")


if __name__ == "__main__":
    main()
