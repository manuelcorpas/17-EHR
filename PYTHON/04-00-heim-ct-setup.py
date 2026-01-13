#!/usr/bin/env python3
"""
04-00-heim-ct-setup.py
======================
HEIM-CT: Clinical Trials Integration - Setup and Schema Validation

Tests AACT database connection and profiles schema for demographic data.
Validates that required tables exist and contain usable fields.

AACT (Aggregate Analysis of ClinicalTrials.gov) is a PostgreSQL database
containing all ClinicalTrials.gov data, updated daily.

INPUT:
    Environment variables:
        AACT_USER     - AACT username (register at https://aact.ctti-clinicaltrials.org)
        AACT_PASSWORD - AACT password
        
OUTPUT:
    DATA/heim_ct_schema_profile.json
    DATA/heim_ct_demographic_categories.csv
    ANALYSIS/HEIM-CT/setup_diagnostic_report.txt

USAGE:
    export AACT_USER='your_username'
    export AACT_PASSWORD='your_password'
    python 04-00-heim-ct-setup.py

REGISTRATION:
    https://aact.ctti-clinicaltrials.org/users/sign_up

VERSION: HEIM-CT v1.0
DATE: 2026-01-13
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

try:
    import psycopg2
    from psycopg2 import sql
except ImportError:
    print("❌ psycopg2 not installed. Run:")
    print("   pip install psycopg2-binary --break-system-packages")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Version metadata
VERSION = "HEIM-CT v1.0"
VERSION_DATE = "2026-01-13"

# Paths
BASE_DIR = Path(__file__).parent.parent if Path(__file__).parent.name == "PYTHON" else Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"
ANALYSIS_DIR = BASE_DIR / "ANALYSIS" / "HEIM-CT"

# AACT connection settings
AACT_HOST = "aact-db.ctti-clinicaltrials.org"
AACT_PORT = 5432
AACT_DATABASE = "aact"

# Priority tables for HEIM-CT analysis
PRIORITY_TABLES = [
    "studies",
    "baseline_measurements",
    "baseline_counts",
    "eligibilities",
    "countries",
    "conditions",
    "browse_conditions",
    "sponsors",
    "result_groups",
    "calculated_values",
    "interventions",
    "facilities"
]


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_aact_connection():
    """Establish connection to AACT PostgreSQL database."""
    user = os.environ.get("AACT_USER")
    password = os.environ.get("AACT_PASSWORD")
    
    if not user or not password:
        raise EnvironmentError(
            "AACT credentials not found. Set environment variables:\n"
            "  export AACT_USER='your_username'\n"
            "  export AACT_PASSWORD='your_password'\n\n"
            "Register at: https://aact.ctti-clinicaltrials.org/users/sign_up"
        )
    
    conn = psycopg2.connect(
        host=AACT_HOST,
        port=AACT_PORT,
        database=AACT_DATABASE,
        user=user,
        password=password,
        connect_timeout=30
    )
    return conn


def get_table_info(conn, table_name: str) -> Dict:
    """Get column information and row count for a table."""
    cur = conn.cursor()
    
    # Get columns
    cur.execute("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'ctgov' AND table_name = %s
        ORDER BY ordinal_position
    """, (table_name,))
    columns = [{"name": r[0], "type": r[1], "nullable": r[2]} for r in cur.fetchall()]
    
    # Get approximate row count
    cur.execute(sql.SQL("""
        SELECT reltuples::bigint 
        FROM pg_class 
        WHERE relname = %s
    """), (table_name,))
    result = cur.fetchone()
    row_count = int(result[0]) if result else 0
    
    cur.close()
    return {"columns": columns, "row_count_approx": row_count}


def profile_priority_tables(conn) -> Dict:
    """Profile all priority tables for HEIM-CT analysis."""
    profile = {}
    
    for table in PRIORITY_TABLES:
        logger.info(f"   Profiling: {table}")
        try:
            info = get_table_info(conn, table)
            profile[table] = {
                "status": "available",
                "columns": info["columns"],
                "row_count_approx": info["row_count_approx"]
            }
        except Exception as e:
            logger.warning(f"   {table}: NOT AVAILABLE - {e}")
            profile[table] = {"status": "unavailable", "error": str(e)}
    
    return profile


def analyze_baseline_measurements(conn) -> Dict:
    """Deep analysis of baseline_measurements table for demographic data."""
    logger.info("   Analyzing demographic categories...")
    cur = conn.cursor()
    
    analysis = {
        "total_rows": 0,
        "studies_with_demographics": 0,
        "category_types": {},
        "race_categories": [],
        "ethnicity_categories": [],
        "sex_categories": [],
        "age_categories": []
    }
    
    # Total rows
    cur.execute("SELECT COUNT(*) FROM ctgov.baseline_measurements")
    analysis["total_rows"] = cur.fetchone()[0]
    
    # Distinct studies
    cur.execute("SELECT COUNT(DISTINCT nct_id) FROM ctgov.baseline_measurements")
    analysis["studies_with_demographics"] = cur.fetchone()[0]
    
    # Category types (titles)
    cur.execute("""
        SELECT LOWER(title), COUNT(*) as cnt
        FROM ctgov.baseline_measurements
        WHERE title IS NOT NULL
        GROUP BY LOWER(title)
        ORDER BY cnt DESC
        LIMIT 100
    """)
    analysis["category_types"] = {r[0]: r[1] for r in cur.fetchall()}
    
    # Race categories
    cur.execute("""
        SELECT DISTINCT category
        FROM ctgov.baseline_measurements
        WHERE (LOWER(title) LIKE '%race%' OR LOWER(title) LIKE '%ethnic%')
          AND category IS NOT NULL
        ORDER BY category
    """)
    analysis["race_categories"] = [r[0] for r in cur.fetchall() if r[0]]
    
    # Sex categories
    cur.execute("""
        SELECT DISTINCT category
        FROM ctgov.baseline_measurements
        WHERE (LOWER(title) LIKE '%sex%' OR LOWER(title) LIKE '%gender%')
          AND category IS NOT NULL
        ORDER BY category
    """)
    analysis["sex_categories"] = [r[0] for r in cur.fetchall() if r[0]]
    
    # Age categories
    cur.execute("""
        SELECT DISTINCT category
        FROM ctgov.baseline_measurements
        WHERE LOWER(title) LIKE '%age%'
          AND category IS NOT NULL
        ORDER BY category
        LIMIT 50
    """)
    analysis["age_categories"] = [r[0] for r in cur.fetchall() if r[0]]
    
    cur.close()
    return analysis


def analyze_conditions_coverage(conn) -> Dict:
    """Analyze conditions table for disease coverage."""
    logger.info("   Analyzing conditions coverage...")
    cur = conn.cursor()
    
    analysis = {
        "total_condition_entries": 0,
        "distinct_conditions": 0,
        "studies_with_conditions": 0,
        "top_conditions": []
    }
    
    cur.execute("SELECT COUNT(*) FROM ctgov.conditions")
    analysis["total_condition_entries"] = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT LOWER(name)) FROM ctgov.conditions WHERE name IS NOT NULL")
    analysis["distinct_conditions"] = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT nct_id) FROM ctgov.conditions")
    analysis["studies_with_conditions"] = cur.fetchone()[0]
    
    cur.execute("""
        SELECT LOWER(name), COUNT(*) as cnt
        FROM ctgov.conditions
        WHERE name IS NOT NULL
        GROUP BY LOWER(name)
        ORDER BY cnt DESC
        LIMIT 50
    """)
    analysis["top_conditions"] = [{"condition": r[0], "count": r[1]} for r in cur.fetchall()]
    
    cur.close()
    return analysis


def analyze_browse_conditions(conn) -> Dict:
    """Analyze browse_conditions for MeSH term coverage."""
    logger.info("   Analyzing MeSH coverage...")
    cur = conn.cursor()
    
    analysis = {
        "total_entries": 0,
        "distinct_mesh_terms": 0,
        "studies_with_mesh": 0,
        "top_mesh_terms": []
    }
    
    cur.execute("SELECT COUNT(*) FROM ctgov.browse_conditions")
    analysis["total_entries"] = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT LOWER(mesh_term)) FROM ctgov.browse_conditions WHERE mesh_term IS NOT NULL")
    analysis["distinct_mesh_terms"] = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT nct_id) FROM ctgov.browse_conditions")
    analysis["studies_with_mesh"] = cur.fetchone()[0]
    
    cur.execute("""
        SELECT mesh_term, COUNT(*) as cnt
        FROM ctgov.browse_conditions
        WHERE mesh_term IS NOT NULL
        GROUP BY mesh_term
        ORDER BY cnt DESC
        LIMIT 50
    """)
    analysis["top_mesh_terms"] = [{"mesh_term": r[0], "count": r[1]} for r in cur.fetchall()]
    
    cur.close()
    return analysis


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def write_diagnostic_report(
    profile: Dict,
    baseline_analysis: Dict,
    conditions_analysis: Dict,
    mesh_analysis: Dict,
    output_path: Path
) -> None:
    """Write human-readable diagnostic report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("HEIM-CT Setup Diagnostic Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Version: {VERSION}\n")
        f.write("=" * 80 + "\n\n")
        
        # Connection status
        f.write("DATABASE CONNECTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Host: {AACT_HOST}\n")
        f.write(f"Database: {AACT_DATABASE}\n")
        f.write("Status: CONNECTED ✓\n\n")
        
        # Table availability
        f.write("TABLE AVAILABILITY\n")
        f.write("-" * 40 + "\n")
        available = sum(1 for t in profile.values() if t.get("status") == "available")
        f.write(f"Available: {available}/{len(PRIORITY_TABLES)}\n\n")
        
        for table, info in profile.items():
            status = info.get("status", "unknown")
            if status == "available":
                rows = info.get("row_count_approx", 0)
                cols = len(info.get("columns", []))
                f.write(f"  ✓ {table:30} ({rows:>12,} rows, {cols:>3} cols)\n")
            else:
                f.write(f"  ✗ {table:30} MISSING\n")
        
        f.write("\n")
        
        # Baseline measurements analysis
        f.write("DEMOGRAPHIC DATA AVAILABILITY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total baseline measurement rows: {baseline_analysis['total_rows']:,}\n")
        f.write(f"Studies with demographic data: {baseline_analysis['studies_with_demographics']:,}\n")
        f.write(f"Race/ethnicity categories: {len(baseline_analysis['race_categories'])}\n")
        f.write(f"Sex/gender categories: {len(baseline_analysis['sex_categories'])}\n")
        f.write(f"Age categories: {len(baseline_analysis['age_categories'])}\n\n")
        
        f.write("Top demographic category titles:\n")
        for title, count in list(baseline_analysis["category_types"].items())[:20]:
            f.write(f"  {title[:50]:50} {count:>10,}\n")
        f.write("\n")
        
        # Conditions coverage
        f.write("CONDITIONS COVERAGE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total condition entries: {conditions_analysis['total_condition_entries']:,}\n")
        f.write(f"Distinct conditions: {conditions_analysis['distinct_conditions']:,}\n")
        f.write(f"Studies with conditions: {conditions_analysis['studies_with_conditions']:,}\n\n")
        
        f.write("Top 20 conditions:\n")
        for item in conditions_analysis["top_conditions"][:20]:
            f.write(f"  {item['condition'][:50]:50} {item['count']:>10,}\n")
        f.write("\n")
        
        # MeSH coverage
        f.write("MeSH TERM COVERAGE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total MeSH entries: {mesh_analysis['total_entries']:,}\n")
        f.write(f"Distinct MeSH terms: {mesh_analysis['distinct_mesh_terms']:,}\n")
        f.write(f"Studies with MeSH: {mesh_analysis['studies_with_mesh']:,}\n\n")
        
        f.write("Top 20 MeSH terms:\n")
        for item in mesh_analysis["top_mesh_terms"][:20]:
            f.write(f"  {item['mesh_term'][:50]:50} {item['count']:>10,}\n")
        f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        if baseline_analysis["studies_with_demographics"] > 10000:
            f.write("✓ Sufficient demographic data for robust analysis\n")
        else:
            f.write("⚠ Limited demographic data; interpret with caution\n")
        
        if conditions_analysis["studies_with_conditions"] > 100000:
            f.write("✓ Strong condition coverage for disease mapping\n")
        else:
            f.write("⚠ Moderate condition coverage\n")
        
        if mesh_analysis["studies_with_mesh"] > 50000:
            f.write("✓ Good MeSH coverage for GBD disease mapping\n")
        else:
            f.write("⚠ Limited MeSH coverage; rely on free-text conditions\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("Setup complete. Proceed to 04-01-heim-ct-fetch.py\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"   Report saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(f"HEIM-CT: Setup and Schema Validation")
    print(f"Version: {VERSION}")
    print("=" * 70)
    
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Test connection
    print(f"\n🔌 Testing AACT database connection...")
    print(f"   Host: {AACT_HOST}")
    
    try:
        conn = get_aact_connection()
        print(f"   ✓ Connection successful!")
    except EnvironmentError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        sys.exit(1)
    
    try:
        # Profile tables
        print(f"\n📊 Profiling priority tables...")
        profile = profile_priority_tables(conn)
        
        # Analyze baseline measurements
        print(f"\n👥 Analyzing demographic data...")
        baseline_analysis = analyze_baseline_measurements(conn)
        print(f"   Total rows: {baseline_analysis['total_rows']:,}")
        print(f"   Studies with demographics: {baseline_analysis['studies_with_demographics']:,}")
        print(f"   Race/ethnicity categories: {len(baseline_analysis['race_categories'])}")
        
        # Analyze conditions
        print(f"\n🏥 Analyzing conditions...")
        conditions_analysis = analyze_conditions_coverage(conn)
        print(f"   Distinct conditions: {conditions_analysis['distinct_conditions']:,}")
        print(f"   Studies with conditions: {conditions_analysis['studies_with_conditions']:,}")
        
        # Analyze MeSH coverage
        print(f"\n🔬 Analyzing MeSH coverage...")
        mesh_analysis = analyze_browse_conditions(conn)
        print(f"   Distinct MeSH terms: {mesh_analysis['distinct_mesh_terms']:,}")
        print(f"   Studies with MeSH: {mesh_analysis['studies_with_mesh']:,}")
        
        # Save schema profile
        print(f"\n💾 Saving outputs...")
        
        schema_output = DATA_DIR / "heim_ct_schema_profile.json"
        with open(schema_output, "w") as f:
            json.dump({
                "generated": datetime.now().isoformat(),
                "version": VERSION,
                "tables": profile,
                "baseline_analysis": baseline_analysis,
                "conditions_analysis": conditions_analysis,
                "mesh_analysis": mesh_analysis
            }, f, indent=2)
        print(f"   {schema_output}")
        
        # Save demographic categories
        if baseline_analysis["race_categories"]:
            demo_records = []
            for cat in baseline_analysis["race_categories"]:
                demo_records.append({"category": cat, "type": "race_ethnicity"})
            for cat in baseline_analysis["sex_categories"]:
                demo_records.append({"category": cat, "type": "sex_gender"})
            
            demo_df = pd.DataFrame(demo_records)
            demo_output = DATA_DIR / "heim_ct_demographic_categories.csv"
            demo_df.to_csv(demo_output, index=False)
            print(f"   {demo_output}")
        
        # Write diagnostic report
        report_output = ANALYSIS_DIR / "setup_diagnostic_report.txt"
        write_diagnostic_report(
            profile, baseline_analysis, conditions_analysis, mesh_analysis,
            report_output
        )
        print(f"   {report_output}")
        
        # Summary
        print(f"\n" + "=" * 70)
        print(f"✅ SETUP COMPLETE")
        print(f"=" * 70)
        
        available = sum(1 for t in profile.values() if t.get("status") == "available")
        print(f"\n📊 Summary:")
        print(f"   Tables available: {available}/{len(PRIORITY_TABLES)}")
        print(f"   Studies with demographics: {baseline_analysis['studies_with_demographics']:,}")
        print(f"   Studies with conditions: {conditions_analysis['studies_with_conditions']:,}")
        print(f"   Studies with MeSH: {mesh_analysis['studies_with_mesh']:,}")
        
        print(f"\n➡️  Next: python 04-01-heim-ct-fetch.py")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
