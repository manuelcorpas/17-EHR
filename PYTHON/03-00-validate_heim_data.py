#!/usr/bin/env python3
"""
HEIM-Biobank v1.0 Comprehensive Data Validity Diagnostic Script
================================================================
Stress-tests BHEM data for inconsistencies, implausible values, and red flags.
Validates BOTH source CSV data (DATA/) AND dashboard JSON data (docs/data/).

USAGE: Run from project root directory:
    python PYTHON/03-00-validate_heim_data.py

Author: HEIM Quality Assurance
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("❌ pandas/numpy required. Install with: pip install pandas numpy")
    sys.exit(1)

# Color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'


class HEIMValidator:
    """Validates HEIM-Biobank data for scientific plausibility and internal consistency."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        
        # Find project root
        self.project_root = self._find_project_root()
        self.data_dir = self.project_root / "DATA"
        self.docs_data_dir = self.project_root / "docs" / "data"
        
        # Known high-publication diseases (from literature - STRICT thresholds)
        self.expected_high_pub_diseases = {
            'Type 2 Diabetes Mellitus': {'min_pubs': 5000, 'category': 'Metabolic'},
            'Ischemic Heart Disease': {'min_pubs': 3000, 'category': 'Cardiovascular'},
            'Stroke': {'min_pubs': 2000, 'category': 'Cardiovascular'},
            'Chronic Kidney Disease': {'min_pubs': 1500, 'category': 'Other NCD'},
            'Alzheimer\'s Disease and Dementia': {'min_pubs': 2000, 'category': 'Neurological'},
            'Breast Cancer': {'min_pubs': 2000, 'category': 'Cancer'},
            'Lung Cancer': {'min_pubs': 1500, 'category': 'Cancer'},
            'COPD': {'min_pubs': 1000, 'category': 'Respiratory'},
            'Asthma': {'min_pubs': 800, 'category': 'Respiratory'},
            'Depressive Disorders': {'min_pubs': 1000, 'category': 'Mental Health'}
        }
        
        # Expected flagship biobank publication ranges
        self.expected_biobank_pubs = {
            'UK Biobank': {'min': 10000, 'max': 20000},
            'eMERGE Network': {'min': 30000, 'max': 60000},
            'FinnGen': {'min': 1500, 'max': 5000},
            'deCODE Genetics': {'min': 3000, 'max': 8000},
            'All of Us Research Program': {'min': 300, 'max': 1500},
            'Estonian Biobank': {'min': 500, 'max': 2000},
            'Million Veteran Program': {'min': 300, 'max': 1000}
        }
    
    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        cwd = Path.cwd()
        
        if cwd.name == "PYTHON" and (cwd.parent / "DATA").exists():
            return cwd.parent
        if (cwd / "DATA").exists() or (cwd / "docs" / "data").exists():
            return cwd
        if (cwd.parent / "DATA").exists():
            return cwd.parent
        
        print(f"{RED}ERROR: Could not find project root directory.{RESET}")
        print(f"Current directory: {cwd}")
        print(f"Expected: PROJECT/DATA/ or PROJECT/docs/data/")
        sys.exit(1)
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find a column from list of candidate names (case-insensitive)."""
        cols_lower = {c.lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate.lower() in cols_lower:
                return cols_lower[candidate.lower()]
        return None
    
    def print_header(self, title: str):
        print(f"\n{BOLD}{BLUE}{'='*70}")
        print(f" {title}")
        print(f"{'='*70}{RESET}")

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_source_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], 
                                        Optional[pd.DataFrame], Optional[Dict]]:
        """Load all BHEM source data files."""
        print(f"\n📂 Loading source data from {self.data_dir}...")
        
        if not self.data_dir.exists():
            print(f"  {YELLOW}DATA/ directory not found - skipping source validation{RESET}")
            return None, None, None, None
        
        diseases, biobanks, publications, metrics = None, None, None, None
        
        # Load disease metrics
        disease_file = self.data_dir / "bhem_disease_metrics.csv"
        if disease_file.exists():
            diseases = pd.read_csv(disease_file)
            print(f"  {GREEN}✓ Loaded {len(diseases)} diseases{RESET}")
            print(f"    Columns: {list(diseases.columns)}")
        else:
            print(f"  {YELLOW}⚠️ bhem_disease_metrics.csv not found{RESET}")
        
        # Load biobank metrics
        biobank_file = self.data_dir / "bhem_biobank_metrics.csv"
        if biobank_file.exists():
            biobanks = pd.read_csv(biobank_file)
            print(f"  {GREEN}✓ Loaded {len(biobanks)} biobanks{RESET}")
            print(f"    Columns: {list(biobanks.columns)}")
        else:
            print(f"  {YELLOW}⚠️ bhem_biobank_metrics.csv not found{RESET}")
        
        # Load publications
        pub_file = self.data_dir / "bhem_publications_mapped.csv"
        if pub_file.exists():
            publications = pd.read_csv(pub_file)
            print(f"  {GREEN}✓ Loaded {len(publications)} publications{RESET}")
            print(f"    Columns: {list(publications.columns)}")
        else:
            print(f"  {YELLOW}⚠️ bhem_publications_mapped.csv not found{RESET}")
        
        # Load JSON metrics
        metrics_file = self.data_dir / "bhem_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            print(f"  {GREEN}✓ Loaded bhem_metrics.json{RESET}")
        
        return diseases, biobanks, publications, metrics
    
    def load_dashboard_data(self) -> Dict:
        """Load dashboard JSON files."""
        print(f"\n📂 Loading dashboard data from {self.docs_data_dir}...")
        
        if not self.docs_data_dir.exists():
            print(f"  {YELLOW}docs/data/ directory not found - skipping dashboard validation{RESET}")
            return {}
        
        dashboard_data = {}
        
        for fname in ['summary', 'biobanks', 'diseases', 'matrix', 'trends', 'themes']:
            fpath = self.docs_data_dir / f"{fname}.json"
            if fpath.exists():
                try:
                    with open(fpath, 'r') as f:
                        dashboard_data[fname] = json.load(f)
                    print(f"  {GREEN}✓ Loaded {fname}.json{RESET}")
                except json.JSONDecodeError as e:
                    self.errors.append(f"Invalid JSON in {fname}.json: {e}")
            else:
                print(f"  {YELLOW}⚠️ {fname}.json not found{RESET}")
        
        return dashboard_data

    # =========================================================================
    # SOURCE DATA VALIDATION CHECKS
    # =========================================================================
    
    def check_zero_publications(self, diseases: pd.DataFrame) -> None:
        """Flag diseases with 0 publications that should have many."""
        print(f"\n🔍 Checking for impossible zero-publication diseases...")
        
        # Find columns
        name_col = self._find_column(diseases, ['disease', 'disease_name', 'name'])
        pubs_col = self._find_column(diseases, ['total_publications', 'publications', 'totalPublications'])
        dalys_col = self._find_column(diseases, ['dalys_millions', 'dalys', 'DALYs'])
        
        if not name_col or not pubs_col:
            self.warnings.append(f"Cannot check zero publications - missing columns")
            return
        
        zero_pub_diseases = diseases[diseases[pubs_col] == 0]
        
        for _, disease in zero_pub_diseases.iterrows():
            disease_name = disease[name_col]
            dalys = disease.get(dalys_col, 0) if dalys_col else 0
            
            # Check if this is a known high-publication disease
            if disease_name in self.expected_high_pub_diseases:
                expected = self.expected_high_pub_diseases[disease_name]
                self.errors.append(
                    f"🚨 CRITICAL: {disease_name} has 0 publications but should have "
                    f"{expected['min_pubs']}+ (one of the most researched conditions globally)"
                )
            elif dalys and dalys > 30:  # High burden disease
                self.errors.append(
                    f"⚠️  {disease_name} has 0 publications despite {dalys}M DALYs burden"
                )
    
    def check_publication_plausibility(self, diseases: pd.DataFrame) -> None:
        """Check if publication counts are plausible given disease burden and research history."""
        print(f"\n🔍 Checking publication count plausibility...")
        
        name_col = self._find_column(diseases, ['disease', 'disease_name', 'name'])
        pubs_col = self._find_column(diseases, ['total_publications', 'publications', 'totalPublications'])
        
        if not name_col or not pubs_col:
            self.warnings.append(f"Cannot check plausibility - missing columns")
            return
        
        for disease_name, expected in self.expected_high_pub_diseases.items():
            disease_row = diseases[diseases[name_col] == disease_name]
            
            if disease_row.empty:
                # Try partial match
                disease_row = diseases[diseases[name_col].str.contains(disease_name.split()[0], case=False, na=False)]
            
            if disease_row.empty:
                self.warnings.append(f"⚠️  Expected disease '{disease_name}' not found in dataset")
                continue
                
            actual_pubs = disease_row.iloc[0][pubs_col]
            min_expected = expected['min_pubs']
            
            if actual_pubs < min_expected:
                deficit_pct = ((min_expected - actual_pubs) / min_expected) * 100
                self.errors.append(
                    f"🚨 {disease_name}: {actual_pubs} publications (expected ≥{min_expected}, "
                    f"{deficit_pct:.0f}% deficit)"
                )
            else:
                self.info.append(f"✓ {disease_name}: {actual_pubs} publications (plausible)")
    
    def check_biobank_publication_ranges(self, biobanks: pd.DataFrame) -> None:
        """Check if flagship biobanks have expected publication volumes."""
        print(f"\n🔍 Checking biobank publication volumes...")
        
        name_col = self._find_column(biobanks, ['biobank', 'biobank_name', 'name', 'Biobank'])
        pubs_col = self._find_column(biobanks, ['total_publications', 'publications', 'totalPublications'])
        
        if not name_col or not pubs_col:
            self.warnings.append(f"Cannot check biobank volumes - missing columns. Available: {list(biobanks.columns)}")
            return
        
        for biobank_name, expected in self.expected_biobank_pubs.items():
            # Flexible matching
            mask = biobanks[name_col].str.contains(biobank_name.split()[0], case=False, na=False)
            biobank_row = biobanks[mask]
            
            if biobank_row.empty:
                continue  # Not all biobanks expected in every dataset
            
            actual_pubs = biobank_row.iloc[0][pubs_col]
            min_exp = expected['min']
            max_exp = expected['max']
            
            if actual_pubs < min_exp:
                self.warnings.append(
                    f"⚠️  {biobank_name}: {actual_pubs} publications (expected {min_exp}-{max_exp})"
                )
            elif actual_pubs > max_exp:
                self.warnings.append(
                    f"⚠️  {biobank_name}: {actual_pubs} publications exceeds expected range "
                    f"({min_exp}-{max_exp}) - possible duplicate counting?"
                )
            else:
                self.info.append(f"✓ {biobank_name}: {actual_pubs} publications (within expected range)")
    
    def check_gap_score_consistency(self, diseases: pd.DataFrame) -> None:
        """Verify gap scores are consistent with burden and publication counts."""
        print(f"\n🔍 Checking gap score consistency...")
        
        name_col = self._find_column(diseases, ['disease', 'disease_name', 'name'])
        pubs_col = self._find_column(diseases, ['total_publications', 'publications'])
        gap_col = self._find_column(diseases, ['gap_score', 'gapScore', 'research_gap_score'])
        dalys_col = self._find_column(diseases, ['dalys_millions', 'dalys'])
        
        if not all([name_col, pubs_col, gap_col]):
            self.warnings.append(f"Cannot check gap scores - missing columns")
            return
        
        for _, disease in diseases.iterrows():
            disease_name = disease[name_col]
            gap_score = disease[gap_col]
            pubs = disease[pubs_col]
            dalys = disease.get(dalys_col, 0) if dalys_col else 0
            
            # Zero publications should always be Critical (gap score ~95)
            if pubs == 0 and gap_score < 90:
                self.errors.append(
                    f"🚨 {disease_name}: 0 publications but gap score only {gap_score} "
                    f"(should be ≥90)"
                )
            
            # Very high publications with high burden should have low gap scores
            if pubs > 1000 and dalys and dalys > 50 and gap_score > 50:
                self.warnings.append(
                    f"⚠️  {disease_name}: {pubs} publications with {dalys}M DALYs "
                    f"but gap score is {gap_score} (seems high)"
                )
    
    def check_biobank_disease_coverage(self, diseases: pd.DataFrame) -> None:
        """Check if biobank coverage makes sense."""
        print(f"\n🔍 Checking biobank-disease coverage...")
        
        name_col = self._find_column(diseases, ['disease', 'disease_name', 'name'])
        pubs_col = self._find_column(diseases, ['total_publications', 'publications'])
        biobanks_col = self._find_column(diseases, ['biobanks_engaged', 'biobanks_count', 'numBiobanks'])
        
        if not all([name_col, pubs_col, biobanks_col]):
            self.warnings.append(f"Cannot check biobank coverage - missing columns")
            return
        
        for _, disease in diseases.iterrows():
            disease_name = disease[name_col]
            pubs = disease[pubs_col]
            biobanks_engaged = disease[biobanks_col]
            
            # Zero publications should mean zero biobanks
            if pubs == 0 and biobanks_engaged > 0:
                self.errors.append(
                    f"🚨 {disease_name}: 0 publications but {biobanks_engaged} biobanks listed"
                )
            
            # High publications should have many biobanks
            if pubs > 1000 and biobanks_engaged < 5:
                self.warnings.append(
                    f"⚠️  {disease_name}: {pubs} publications but only {biobanks_engaged} biobanks "
                    f"(expected more engagement)"
                )
    
    def check_mesh_mapping_coverage(self, publications: pd.DataFrame) -> None:
        """Check MeSH mapping quality."""
        print(f"\n🔍 Checking MeSH mapping coverage...")
        
        disease_col = self._find_column(publications, ['disease', 'disease_mapped', 'mapped_disease'])
        
        if not disease_col:
            self.warnings.append(f"Cannot check mapping - no disease column found")
            return
        
        total_pubs = len(publications)
        mapped_pubs = publications[disease_col].notna().sum()
        unmapped_pubs = total_pubs - mapped_pubs
        
        mapping_rate = (mapped_pubs / total_pubs) * 100 if total_pubs > 0 else 0
        
        self.info.append(f"✓ Mapped {mapped_pubs}/{total_pubs} publications ({mapping_rate:.1f}%)")
        
        if mapping_rate < 70:
            self.warnings.append(
                f"⚠️  Low mapping rate: {mapping_rate:.1f}% (target: ≥70%)"
            )
        
        # Check for diseases with suspiciously few publications
        disease_counts = publications[disease_col].value_counts()
        for disease, count in disease_counts.items():
            if disease in self.expected_high_pub_diseases and count < 100:
                self.errors.append(
                    f"🚨 {disease}: Only {count} mapped publications in raw data "
                    f"(mapping likely incomplete)"
                )
    
    def check_equity_ratio_sanity(self, metrics: Dict) -> None:
        """Check if equity ratios are plausible."""
        print(f"\n🔍 Checking equity ratio plausibility...")
        
        if not metrics:
            return
        
        equity_ratio = metrics.get('equity_ratio') or metrics.get('equityRatio')
        
        if equity_ratio:
            # Equity ratio should be >1 (HIC bias expected) but not absurdly high
            if equity_ratio < 1:
                self.warnings.append(
                    f"⚠️  Equity ratio {equity_ratio:.1f} < 1 (unexpected: suggests Global South bias)"
                )
            elif equity_ratio > 100:
                self.errors.append(
                    f"🚨 Equity ratio {equity_ratio:.1f} > 100 (implausibly high - calculation error?)"
                )
            else:
                self.info.append(f"✓ Equity ratio: {equity_ratio:.1f}× (plausible HIC bias)")
    
    def check_summary_consistency(self, diseases: pd.DataFrame, biobanks: pd.DataFrame, 
                                  publications: pd.DataFrame, metrics: Dict) -> None:
        """Check if summary metrics match detail data."""
        print(f"\n🔍 Checking summary metric consistency...")
        
        if not metrics:
            return
        
        pmid_col = self._find_column(publications, ['pmid', 'PMID', 'pubmed_id'])
        gap_cat_col = self._find_column(diseases, ['gap_category', 'gapCategory', 'gap_severity'])
        
        # Total publications
        if pmid_col:
            total_pubs_from_data = publications[pmid_col].nunique()
            total_pubs_reported = metrics.get('total_publications') or metrics.get('totalPublications', 0)
            
            if abs(total_pubs_from_data - total_pubs_reported) > 100:
                self.warnings.append(
                    f"⚠️  Publication count mismatch: {total_pubs_reported} reported, "
                    f"{total_pubs_from_data} in raw data"
                )
        
        # Critical gap count
        if gap_cat_col:
            critical_gaps_actual = len(diseases[diseases[gap_cat_col] == 'Critical'])
            critical_gaps_reported = metrics.get('critical_gap_diseases') or metrics.get('criticalGaps', 0)
            
            if critical_gaps_actual != critical_gaps_reported:
                self.warnings.append(
                    f"⚠️  Critical gap count mismatch: {critical_gaps_reported} reported, "
                    f"{critical_gaps_actual} in disease data"
                )

    # =========================================================================
    # DASHBOARD DATA VALIDATION
    # =========================================================================
    
    def check_dashboard_diseases(self, data: Dict) -> None:
        """Validate dashboard diseases.json."""
        print(f"\n🔍 Checking dashboard disease data...")
        
        diseases = data.get('diseases', [])
        zero_pub_diseases = []
        
        for d in diseases:
            name = d.get('name', 'Unknown')
            pubs = d.get('research', {}).get('globalPublications', 0)
            dalys = d.get('burden', {}).get('dalysMillions', 0)
            
            if pubs == 0:
                zero_pub_diseases.append((name, dalys))
                
                if name in self.expected_high_pub_diseases:
                    self.errors.append(f"🚨 Dashboard: {name} shows 0 publications")
        
        if zero_pub_diseases:
            print(f"  {RED}Zero-publication diseases in dashboard:{RESET}")
            for name, dalys in sorted(zero_pub_diseases, key=lambda x: -x[1])[:5]:
                print(f"    • {name}: {dalys}M DALYs")
    
    def check_dashboard_matrix(self, data: Dict, all_data: Dict) -> None:
        """Validate dashboard matrix.json."""
        print(f"\n🔍 Checking dashboard matrix data...")
        
        biobanks = data.get('biobanks', [])
        diseases = data.get('diseases', [])
        matrix = data.get('matrix', {})
        values = matrix.get('values', [])
        
        print(f"    Matrix dimensions: {len(biobanks)} biobanks × {len(diseases)} diseases")
        
        # Check dimensions
        if len(values) != len(biobanks):
            self.errors.append(f"Matrix rows ({len(values)}) != biobanks ({len(biobanks)})")
        
        # Calculate totals
        matrix_total = sum(sum(row) for row in values)
        
        # Compare with biobanks.json
        if 'biobanks' in all_data:
            biobank_total = sum(
                b.get('stats', {}).get('totalPublications', 0) 
                for b in all_data['biobanks'].get('biobanks', [])
            )
            
            if matrix_total != biobank_total:
                coverage = 100 * matrix_total / biobank_total if biobank_total > 0 else 0
                self.warnings.append(f"Matrix total ({matrix_total:,}) != biobanks ({biobank_total:,})")
                self.warnings.append(f"Matrix covers only {coverage:.1f}% of publications")
                
                if coverage < 50:
                    self.errors.append(f"🚨 Matrix severely incomplete ({coverage:.1f}% coverage)")
        
        # Check for zero-sum disease columns
        if values and diseases:
            disease_totals = [0] * len(diseases)
            for row in values:
                for di, val in enumerate(row):
                    if di < len(disease_totals):
                        disease_totals[di] += val
            
            zero_diseases = [diseases[i].get('name', f'Disease {i}') 
                            for i, total in enumerate(disease_totals) if total == 0]
            
            if zero_diseases:
                print(f"  {RED}Diseases with 0 in matrix:{RESET}")
                for name in zero_diseases[:5]:
                    self.errors.append(f"🚨 Matrix: {name} has 0 publications")

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================
    
    def run(self) -> int:
        """Run all validation checks and return exit code."""
        print(f"\n{BOLD}{CYAN}{'='*70}")
        print("HEIM-BIOBANK v1.0 DATA VALIDITY DIAGNOSTIC")
        print(f"{'='*70}{RESET}")
        print(f"Project root: {self.project_root}")
        
        # Part 1: Source data validation
        self.print_header("PART 1: SOURCE DATA VALIDATION")
        diseases, biobanks, publications, metrics = self.load_source_data()
        
        if diseases is not None:
            self.check_zero_publications(diseases)
            self.check_publication_plausibility(diseases)
            self.check_gap_score_consistency(diseases)
            self.check_biobank_disease_coverage(diseases)
        
        if biobanks is not None:
            self.check_biobank_publication_ranges(biobanks)
        
        if publications is not None:
            self.check_mesh_mapping_coverage(publications)
        
        if metrics:
            self.check_equity_ratio_sanity(metrics)
        
        if all([diseases is not None, biobanks is not None, publications is not None, metrics]):
            self.check_summary_consistency(diseases, biobanks, publications, metrics)
        
        # Part 2: Dashboard data validation
        self.print_header("PART 2: DASHBOARD DATA VALIDATION")
        dashboard_data = self.load_dashboard_data()
        
        if 'diseases' in dashboard_data:
            self.check_dashboard_diseases(dashboard_data['diseases'])
        
        if 'matrix' in dashboard_data:
            self.check_dashboard_matrix(dashboard_data['matrix'], dashboard_data)
        
        # Generate report
        return self.generate_report()
    
    def generate_report(self) -> int:
        """Generate diagnostic report and return exit code."""
        self.print_header("DIAGNOSTIC REPORT SUMMARY")
        
        print(f"\n{RED}🚨 ERRORS ({len(self.errors)}):{RESET}")
        if self.errors:
            for error in self.errors:
                print(f"  {error}")
        else:
            print(f"  {GREEN}✓ No errors found{RESET}")
        
        print(f"\n{YELLOW}⚠️  WARNINGS ({len(self.warnings)}):{RESET}")
        if self.warnings:
            for warning in self.warnings:
                print(f"  {warning}")
        else:
            print(f"  {GREEN}✓ No warnings{RESET}")
        
        print(f"\n{GREEN}✓ INFO ({len(self.info)}):{RESET}")
        for info in self.info[:10]:
            print(f"  {info}")
        if len(self.info) > 10:
            print(f"  ... and {len(self.info) - 10} more")
        
        # Severity assessment
        print(f"\n{BOLD}{'='*70}")
        print("SEVERITY ASSESSMENT")
        print(f"{'='*70}{RESET}")
        
        if len(self.errors) == 0 and len(self.warnings) == 0:
            print(f"\n  {GREEN}{BOLD}✅ ALL VALIDATIONS PASSED{RESET}")
            print(f"  Data appears scientifically plausible and internally consistent.")
            return 0
        
        elif len(self.errors) == 0:
            print(f"\n  {YELLOW}{BOLD}⚠️  MINOR ISSUES DETECTED{RESET}")
            print(f"  Data is usable but has {len(self.warnings)} warnings to review.")
            return 1
        
        else:
            print(f"\n  {RED}{BOLD}❌ CRITICAL ISSUES DETECTED{RESET}")
            print(f"  Found {len(self.errors)} errors and {len(self.warnings)} warnings.")
            print(f"\n  {BOLD}RECOMMENDED ACTIONS:{RESET}")
            
            if any('0 publications' in str(e) for e in self.errors):
                print(f"    1. {RED}Review MeSH-to-disease mapping - major diseases show 0 publications{RESET}")
                print(f"       This suggests the mapping dictionary is incomplete or matching failed.")
            
            if any('matrix' in str(e).lower() for e in self.errors):
                print(f"    2. {RED}Regenerate matrix.json - publications not properly mapped to diseases{RESET}")
            
            if any('gap score' in str(e).lower() for e in self.errors):
                print(f"    3. {RED}Recalculate gap scores - inconsistent with publication counts{RESET}")
            
            return 2


def main():
    validator = HEIMValidator()
    exit_code = validator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()