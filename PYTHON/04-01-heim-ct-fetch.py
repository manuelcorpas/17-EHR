#!/usr/bin/env python3
"""
04-01-heim-ct-fetch.py
======================
HEIM-CT: Clinical Trials Data Fetcher

Extracts clinical trial data from AACT PostgreSQL database.
Fetches studies, demographics, conditions, countries, and sponsors.

INPUT:
    Environment variables:
        AACT_USER     - AACT username
        AACT_PASSWORD - AACT password
        
OUTPUT:
    DATA/heim_ct_studies.csv
    DATA/heim_ct_baseline.csv
    DATA/heim_ct_conditions.csv
    DATA/heim_ct_countries.csv
    DATA/heim_ct_sponsors.csv
    DATA/heim_ct_fetch_manifest.json

USAGE:
    python 04-01-heim-ct-fetch.py [--start-year 2010] [--end-year 2024]

VERSION: HEIM-CT v1.0
DATE: 2026-01-13
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

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

VERSION = "HEIM-CT v1.0"
VERSION_DATE = "2026-01-13"

# Paths
BASE_DIR = Path(__file__).parent.parent if Path(__file__).parent.name == "PYTHON" else Path(__file__).parent
DATA_DIR = BASE_DIR / "DATA"

# AACT connection
AACT_HOST = "aact-db.ctti-clinicaltrials.org"
AACT_PORT = 5432
AACT_DATABASE = "aact"

# Default year range
DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR = 2024

# Chunked fetching
CHUNK_SIZE = 50000


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
            "  export AACT_PASSWORD='your_password'"
        )
    
    conn = psycopg2.connect(
        host=AACT_HOST,
        port=AACT_PORT,
        database=AACT_DATABASE,
        user=user,
        password=password,
        connect_timeout=60
    )
    return conn


def fetch_studies(conn, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch study records with year filtering."""
    logger.info(f"   Fetching studies ({start_year}-{end_year})...")
    
    query = """
        SELECT 
            nct_id,
            brief_title,
            official_title,
            overall_status,
            phase,
            study_type,
            enrollment,
            enrollment_type,
            start_date,
            completion_date,
            primary_completion_date,
            results_first_posted_date,
            last_update_posted_date,
            study_first_posted_date,
            EXTRACT(YEAR FROM start_date) as start_year,
            EXTRACT(YEAR FROM study_first_posted_date) as posted_year,
            source,
            number_of_arms,
            number_of_groups,
            has_dmc,
            is_fda_regulated_drug,
            is_fda_regulated_device,
            is_us_export
        FROM ctgov.studies
        WHERE (
            EXTRACT(YEAR FROM start_date) BETWEEN %s AND %s
            OR EXTRACT(YEAR FROM study_first_posted_date) BETWEEN %s AND %s
        )
    """
    
    df = pd.read_sql(query, conn, params=(start_year, end_year, start_year, end_year))
    logger.info(f"      Fetched {len(df):,} studies")
    return df


def fetch_baseline_measurements(conn, nct_ids: List[str]) -> pd.DataFrame:
    """Fetch baseline demographic measurements for specified studies."""
    logger.info(f"   Fetching baseline measurements...")
    
    if not nct_ids:
        return pd.DataFrame()
    
    # Chunked fetching for large NCT ID lists
    all_data = []
    
    for i in range(0, len(nct_ids), CHUNK_SIZE):
        chunk = nct_ids[i:i+CHUNK_SIZE]
        
        query = """
            SELECT 
                nct_id,
                result_group_id,
                ctgov_group_code,
                title,
                description,
                units,
                param_type,
                param_value,
                param_value_num,
                dispersion_type,
                dispersion_value,
                dispersion_value_num,
                explanation_of_na,
                category,
                classification
            FROM ctgov.baseline_measurements
            WHERE nct_id = ANY(%s)
        """
        
        chunk_df = pd.read_sql(query, conn, params=(chunk,))
        all_data.append(chunk_df)
        
        if len(nct_ids) > CHUNK_SIZE:
            logger.info(f"      Chunk {i//CHUNK_SIZE + 1}: {len(chunk_df):,} rows")
    
    df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    logger.info(f"      Fetched {len(df):,} baseline measurements")
    return df


def fetch_conditions(conn, nct_ids: List[str]) -> pd.DataFrame:
    """Fetch condition names (free text)."""
    logger.info(f"   Fetching conditions...")
    
    if not nct_ids:
        return pd.DataFrame()
    
    all_data = []
    
    for i in range(0, len(nct_ids), CHUNK_SIZE):
        chunk = nct_ids[i:i+CHUNK_SIZE]
        
        query = """
            SELECT nct_id, name, downcase_name
            FROM ctgov.conditions
            WHERE nct_id = ANY(%s)
        """
        
        chunk_df = pd.read_sql(query, conn, params=(chunk,))
        all_data.append(chunk_df)
    
    df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    logger.info(f"      Fetched {len(df):,} condition entries")
    return df


def fetch_browse_conditions(conn, nct_ids: List[str]) -> pd.DataFrame:
    """Fetch MeSH-mapped conditions."""
    logger.info(f"   Fetching MeSH conditions...")
    
    if not nct_ids:
        return pd.DataFrame()
    
    all_data = []
    
    for i in range(0, len(nct_ids), CHUNK_SIZE):
        chunk = nct_ids[i:i+CHUNK_SIZE]
        
        query = """
            SELECT nct_id, mesh_term, downcase_mesh_term, mesh_type
            FROM ctgov.browse_conditions
            WHERE nct_id = ANY(%s)
        """
        
        chunk_df = pd.read_sql(query, conn, params=(chunk,))
        all_data.append(chunk_df)
    
    df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    logger.info(f"      Fetched {len(df):,} MeSH condition entries")
    return df


def fetch_countries(conn, nct_ids: List[str]) -> pd.DataFrame:
    """Fetch country data for studies."""
    logger.info(f"   Fetching countries...")
    
    if not nct_ids:
        return pd.DataFrame()
    
    all_data = []
    
    for i in range(0, len(nct_ids), CHUNK_SIZE):
        chunk = nct_ids[i:i+CHUNK_SIZE]
        
        query = """
            SELECT nct_id, name, removed
            FROM ctgov.countries
            WHERE nct_id = ANY(%s)
        """
        
        chunk_df = pd.read_sql(query, conn, params=(chunk,))
        all_data.append(chunk_df)
    
    df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    logger.info(f"      Fetched {len(df):,} country entries")
    return df


def fetch_sponsors(conn, nct_ids: List[str]) -> pd.DataFrame:
    """Fetch sponsor information."""
    logger.info(f"   Fetching sponsors...")
    
    if not nct_ids:
        return pd.DataFrame()
    
    all_data = []
    
    for i in range(0, len(nct_ids), CHUNK_SIZE):
        chunk = nct_ids[i:i+CHUNK_SIZE]
        
        query = """
            SELECT nct_id, agency_class, lead_or_collaborator, name
            FROM ctgov.sponsors
            WHERE nct_id = ANY(%s)
        """
        
        chunk_df = pd.read_sql(query, conn, params=(chunk,))
        all_data.append(chunk_df)
    
    df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    logger.info(f"      Fetched {len(df):,} sponsor entries")
    return df


def fetch_eligibilities(conn, nct_ids: List[str]) -> pd.DataFrame:
    """Fetch eligibility criteria."""
    logger.info(f"   Fetching eligibility criteria...")
    
    if not nct_ids:
        return pd.DataFrame()
    
    all_data = []
    
    for i in range(0, len(nct_ids), CHUNK_SIZE):
        chunk = nct_ids[i:i+CHUNK_SIZE]
        
        query = """
            SELECT 
                nct_id,
                sampling_method,
                gender,
                gender_based,
                gender_description,
                minimum_age,
                maximum_age,
                healthy_volunteers,
                population,
                criteria
            FROM ctgov.eligibilities
            WHERE nct_id = ANY(%s)
        """
        
        chunk_df = pd.read_sql(query, conn, params=(chunk,))
        all_data.append(chunk_df)
    
    df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    logger.info(f"      Fetched {len(df):,} eligibility entries")
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="HEIM-CT: Fetch clinical trial data from AACT")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR,
                        help=f"Start year (default: {DEFAULT_START_YEAR})")
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR,
                        help=f"End year (default: {DEFAULT_END_YEAR})")
    parser.add_argument("--interventional-only", action="store_true",
                        help="Only fetch interventional studies")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"HEIM-CT: Clinical Trials Data Fetcher")
    print(f"Version: {VERSION}")
    print(f"Year range: {args.start_year}-{args.end_year}")
    print("=" * 70)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    
    # Connect
    print(f"\n🔌 Connecting to AACT database...")
    try:
        conn = get_aact_connection()
        print(f"   ✓ Connected to {AACT_HOST}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        sys.exit(1)
    
    manifest = {
        "version": VERSION,
        "generated": datetime.now().isoformat(),
        "year_range": [args.start_year, args.end_year],
        "tables": {}
    }
    
    try:
        # Fetch studies
        print(f"\n📊 Fetching data...")
        df_studies = fetch_studies(conn, args.start_year, args.end_year)
        
        if args.interventional_only:
            df_studies = df_studies[df_studies["study_type"] == "Interventional"]
            logger.info(f"   Filtered to {len(df_studies):,} interventional studies")
        
        nct_ids = df_studies["nct_id"].tolist()
        
        # Fetch related tables
        df_baseline = fetch_baseline_measurements(conn, nct_ids)
        df_conditions = fetch_conditions(conn, nct_ids)
        df_mesh = fetch_browse_conditions(conn, nct_ids)
        df_countries = fetch_countries(conn, nct_ids)
        df_sponsors = fetch_sponsors(conn, nct_ids)
        df_eligibilities = fetch_eligibilities(conn, nct_ids)
        
        # Save outputs
        print(f"\n💾 Saving outputs...")
        
        outputs = [
            ("heim_ct_studies.csv", df_studies),
            ("heim_ct_baseline.csv", df_baseline),
            ("heim_ct_conditions.csv", df_conditions),
            ("heim_ct_mesh_conditions.csv", df_mesh),
            ("heim_ct_countries.csv", df_countries),
            ("heim_ct_sponsors.csv", df_sponsors),
            ("heim_ct_eligibilities.csv", df_eligibilities),
        ]
        
        for filename, df in outputs:
            filepath = DATA_DIR / filename
            df.to_csv(filepath, index=False)
            manifest["tables"][filename] = {
                "rows": len(df),
                "columns": list(df.columns)
            }
            print(f"   {filepath.name}: {len(df):,} rows")
        
        # Save manifest
        manifest_path = DATA_DIR / "heim_ct_fetch_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"   {manifest_path.name}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Summary
        print(f"\n" + "=" * 70)
        print(f"✅ FETCH COMPLETE")
        print(f"=" * 70)
        
        print(f"\n📊 Summary:")
        print(f"   Studies: {len(df_studies):,}")
        print(f"   Baseline measurements: {len(df_baseline):,}")
        print(f"   Conditions (free text): {len(df_conditions):,}")
        print(f"   Conditions (MeSH): {len(df_mesh):,}")
        print(f"   Countries: {len(df_countries):,}")
        print(f"   Sponsors: {len(df_sponsors):,}")
        print(f"   Eligibilities: {len(df_eligibilities):,}")
        print(f"   Elapsed: {elapsed:.1f}s")
        
        # Studies with demographics
        studies_with_demo = df_baseline["nct_id"].nunique() if len(df_baseline) > 0 else 0
        demo_rate = 100 * studies_with_demo / len(df_studies) if len(df_studies) > 0 else 0
        print(f"\n   Studies with demographic data: {studies_with_demo:,} ({demo_rate:.1f}%)")
        
        print(f"\n➡️  Next: python 04-02-heim-ct-map-diseases.py")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
