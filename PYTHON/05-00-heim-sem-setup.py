#!/usr/bin/env python3
"""
HEIM SEMANTIC INTEGRATION - SETUP & VALIDATION
===============================================

Validates environment and creates directory structure for semantic pipeline.

VALIDATES:
- Python version ≥3.11
- PyTorch ≥2.0 with MPS backend (Apple Silicon)
- Required packages with minimum versions
- Disk space ≥100GB free
- PubMedBERT model loads correctly
- Test embedding matches expected dimensions
- HDF5 read/write operations work

CREATES:
- DATA/05-SEMANTIC/ directory structure
- DATA/05-SEMANTIC/.setup_complete marker
- ANALYSIS/05-XX directories
- LOGS/ directory

EXIT CODES:
- 0: All checks passed
- 1: Critical failure (cannot proceed)
- 2: Warning (can proceed with caution)

USAGE:
    python 05-00-heim-sem-setup.py
    python 05-00-heim-sem-setup.py --skip-model  # Skip model download (for testing)
"""

import os
import sys
import json
import shutil
import hashlib
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "DATA"
ANALYSIS_DIR = BASE_DIR / "ANALYSIS"
LOGS_DIR = BASE_DIR / "LOGS"

# Semantic pipeline directories
SEMANTIC_DIR = DATA_DIR / "05-SEMANTIC"
PUBMED_RAW_DIR = SEMANTIC_DIR / "PUBMED-RAW"
EMBEDDINGS_DIR = SEMANTIC_DIR / "EMBEDDINGS"
CHECKPOINTS_DIR = SEMANTIC_DIR / "CHECKPOINTS"

# Analysis output directories
ANALYSIS_DIRS = [
    ANALYSIS_DIR / "05-01-SEMANTIC-QUALITY",
    ANALYSIS_DIR / "05-02-EMBEDDING-VALIDATION",
    ANALYSIS_DIR / "05-03-SEMANTIC-METRICS",
    ANALYSIS_DIR / "05-04-HEIM-SEM-FIGURES",
]

# Marker file
SETUP_COMPLETE_MARKER = SEMANTIC_DIR / ".setup_complete"

# Minimum requirements
MIN_PYTHON_VERSION = (3, 11)
MIN_DISK_SPACE_GB = 100
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Required packages with minimum versions
REQUIRED_PACKAGES = {
    "torch": "2.0.0",
    "transformers": "4.30.0",
    "pandas": "2.0.0",
    "numpy": "1.24.0",
    "h5py": "3.9.0",
    "scikit-learn": "1.3.0",
    "umap-learn": "0.5.0",
    "matplotlib": "3.7.0",
    "seaborn": "0.12.0",
    "tqdm": "4.65.0",
    "biopython": "1.81",
    "requests": "2.31.0",
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(title: str):
    """Print formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)

def print_check(name: str, passed: bool, detail: str = ""):
    """Print check result with status indicator."""
    status = "✓ PASS" if passed else "✗ FAIL"
    color_start = "\033[92m" if passed else "\033[91m"
    color_end = "\033[0m"

    detail_str = f" ({detail})" if detail else ""
    print(f"  {color_start}{status}{color_end}  {name}{detail_str}")

def print_warning(message: str):
    """Print warning message."""
    print(f"  \033[93m⚠ WARN\033[0m  {message}")

def print_info(message: str):
    """Print info message."""
    print(f"  \033[94mℹ INFO\033[0m  {message}")

def get_disk_space_gb(path: Path) -> float:
    """Get available disk space in GB."""
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)

def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string to tuple for comparison."""
    # Handle versions like "2.0.0+cu118" or "1.24.3"
    clean_version = version_str.split("+")[0].split("rc")[0].split("a")[0].split("b")[0]
    parts = []
    for part in clean_version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts) if parts else (0,)

def version_gte(installed: str, required: str) -> bool:
    """Check if installed version >= required version."""
    return parse_version(installed) >= parse_version(required)

# =============================================================================
# VALIDATION CHECKS
# =============================================================================

def check_python_version() -> Tuple[bool, str]:
    """Check Python version meets minimum requirement."""
    current = sys.version_info[:2]
    passed = current >= MIN_PYTHON_VERSION
    detail = f"{current[0]}.{current[1]} >= {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"
    return passed, detail

def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space."""
    available = get_disk_space_gb(BASE_DIR)
    passed = available >= MIN_DISK_SPACE_GB
    detail = f"{available:.1f}GB available, {MIN_DISK_SPACE_GB}GB required"
    return passed, detail

def check_package(package_name: str, min_version: str) -> Tuple[bool, str]:
    """Check if package is installed with minimum version."""
    try:
        if package_name == "sklearn":
            import sklearn
            installed = sklearn.__version__
        elif package_name == "scikit-learn":
            import sklearn
            installed = sklearn.__version__
        elif package_name == "umap-learn":
            import umap
            installed = umap.__version__
        elif package_name == "biopython":
            import Bio
            installed = Bio.__version__
        else:
            module = __import__(package_name.replace("-", "_"))
            installed = module.__version__

        passed = version_gte(installed, min_version)
        detail = f"{installed} >= {min_version}"
        return passed, detail
    except ImportError:
        return False, "not installed"
    except AttributeError:
        return True, "installed (version unknown)"

def check_pytorch_mps() -> Tuple[bool, str, bool]:
    """Check PyTorch MPS (Metal Performance Shaders) availability."""
    try:
        import torch

        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()

        if mps_available and mps_built:
            return True, f"PyTorch {torch.__version__} with MPS", True
        elif mps_built:
            return True, f"PyTorch {torch.__version__} (MPS built but not available)", False
        else:
            return True, f"PyTorch {torch.__version__} (MPS not built, will use CPU)", False
    except ImportError:
        return False, "PyTorch not installed", False

def check_model_loading(skip: bool = False) -> Tuple[bool, str]:
    """Check if PubMedBERT model can be loaded."""
    if skip:
        return True, "skipped (--skip-model flag)"

    try:
        print_info("Downloading/loading PubMedBERT model (may take a few minutes)...")

        from transformers import AutoTokenizer, AutoModel
        import torch

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)

        # Test tokenization
        test_text = "Malaria is a life-threatening disease caused by parasites."
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)

        # Test forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Check output shape
        embedding = outputs.last_hidden_state.mean(dim=1)
        if embedding.shape[1] != 768:
            return False, f"unexpected embedding dimension: {embedding.shape[1]}"

        # Cleanup
        del model, tokenizer, inputs, outputs, embedding
        import gc
        gc.collect()

        return True, "model loads and produces 768-dim embeddings"
    except Exception as e:
        return False, str(e)

def check_hdf5_operations() -> Tuple[bool, str]:
    """Check HDF5 read/write operations work."""
    try:
        import h5py
        import numpy as np
        import tempfile

        # Create test file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            test_path = f.name

        # Write test data
        test_data = np.random.rand(100, 768).astype(np.float32)
        with h5py.File(test_path, 'w') as f:
            f.create_dataset('embeddings', data=test_data, compression='gzip')

        # Read and verify
        with h5py.File(test_path, 'r') as f:
            loaded = f['embeddings'][:]

        # Cleanup
        os.unlink(test_path)

        if np.allclose(test_data, loaded):
            return True, "read/write with compression verified"
        else:
            return False, "data mismatch after read"
    except Exception as e:
        return False, str(e)

def check_ncbi_api() -> Tuple[bool, str]:
    """Check NCBI Entrez API connectivity."""
    try:
        from Bio import Entrez

        Entrez.email = "test@example.com"
        Entrez.tool = "HEIM-Semantic-Setup"

        # Simple search to test connectivity
        handle = Entrez.esearch(db="pubmed", term="malaria[Title]", retmax=1)
        record = Entrez.read(handle)
        handle.close()

        if int(record["Count"]) > 0:
            return True, "connected, API functional"
        else:
            return False, "connected but no results"
    except Exception as e:
        return False, f"connection failed: {str(e)[:50]}"

# =============================================================================
# DIRECTORY CREATION
# =============================================================================

def create_directories() -> List[str]:
    """Create all required directories."""
    created = []

    directories = [
        SEMANTIC_DIR,
        PUBMED_RAW_DIR,
        EMBEDDINGS_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
    ] + ANALYSIS_DIRS

    for dir_path in directories:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created.append(str(dir_path.relative_to(BASE_DIR)))

    return created

# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_validation(skip_model: bool = False) -> int:
    """Run all validation checks and return exit code."""

    print_header("HEIM SEMANTIC PIPELINE - ENVIRONMENT VALIDATION")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Base Directory: {BASE_DIR}")
    print(f"  Platform: {platform.system()} {platform.machine()}")

    all_passed = True
    warnings = []
    mps_available = False

    # Section 1: System Requirements
    print_header("1. SYSTEM REQUIREMENTS")

    passed, detail = check_python_version()
    print_check("Python version", passed, detail)
    all_passed = all_passed and passed

    passed, detail = check_disk_space()
    print_check("Disk space", passed, detail)
    all_passed = all_passed and passed

    # Section 2: Python Packages
    print_header("2. PYTHON PACKAGES")

    for package, min_version in REQUIRED_PACKAGES.items():
        passed, detail = check_package(package, min_version)
        print_check(f"{package}", passed, detail)
        all_passed = all_passed and passed

    # Section 3: PyTorch & MPS
    print_header("3. PYTORCH & MPS BACKEND")

    passed, detail, mps_available = check_pytorch_mps()
    print_check("PyTorch with MPS", passed, detail)
    if passed and not mps_available:
        print_warning("MPS not available - will fall back to CPU (slower)")
        warnings.append("MPS backend not available")
    all_passed = all_passed and passed

    # Section 4: Model Loading
    print_header("4. PUBMEDBERT MODEL")

    passed, detail = check_model_loading(skip=skip_model)
    print_check("PubMedBERT loading", passed, detail)
    all_passed = all_passed and passed

    # Section 5: HDF5 Operations
    print_header("5. HDF5 STORAGE")

    passed, detail = check_hdf5_operations()
    print_check("HDF5 read/write", passed, detail)
    all_passed = all_passed and passed

    # Section 6: NCBI API
    print_header("6. NCBI API CONNECTIVITY")

    passed, detail = check_ncbi_api()
    print_check("NCBI Entrez API", passed, detail)
    if not passed:
        print_warning("NCBI API check failed - may be temporary network issue")
        warnings.append("NCBI API connectivity issue")
    # Don't fail on API check - could be temporary

    # Section 7: Directory Creation
    print_header("7. DIRECTORY STRUCTURE")

    created = create_directories()
    if created:
        print_info(f"Created {len(created)} directories:")
        for d in created:
            print(f"      + {d}")
    else:
        print_info("All directories already exist")
    print_check("Directory structure", True)

    # Summary
    print_header("VALIDATION SUMMARY")

    if all_passed:
        # Write marker file
        marker_data = {
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": f"{platform.system()} {platform.machine()}",
            "mps_available": mps_available,
            "model_name": MODEL_NAME,
            "packages": {pkg: check_package(pkg, ver)[1] for pkg, ver in REQUIRED_PACKAGES.items()},
            "warnings": warnings,
        }

        with open(SETUP_COMPLETE_MARKER, 'w') as f:
            json.dump(marker_data, f, indent=2)

        print(f"\n  \033[92m✓ ALL CHECKS PASSED\033[0m")
        print(f"  Marker file created: {SETUP_COMPLETE_MARKER.relative_to(BASE_DIR)}")

        if warnings:
            print(f"\n  Warnings ({len(warnings)}):")
            for w in warnings:
                print(f"    - {w}")

        print(f"\n  Ready to proceed with: python 05-01-heim-sem-fetch.py")
        return 0
    else:
        print(f"\n  \033[91m✗ VALIDATION FAILED\033[0m")
        print(f"  Please fix the failed checks above before proceeding.")
        return 1

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="HEIM Semantic Pipeline - Setup & Validation"
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model download/loading check (for testing)"
    )

    args = parser.parse_args()

    exit_code = run_validation(skip_model=args.skip_model)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
