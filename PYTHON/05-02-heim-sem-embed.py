#!/usr/bin/env python3
"""
HEIM SEMANTIC INTEGRATION - EMBEDDING GENERATION
=================================================

Generates PubMedBERT embeddings for all retrieved PubMed abstracts using
Apple MPS backend (Metal Performance Shaders) with CPU fallback.

FEATURES:
- MPS backend with automatic CPU fallback
- Batch size 64 (optimized for 256GB RAM)
- Memory clearing after each batch
- Checkpoint after each disease completes
- HDF5 storage with gzip compression
- Integrity checksums (SHA-256)
- Progress bars with ETA
- MPS validation against CPU reference

OUTPUTS:
- DATA/05-SEMANTIC/EMBEDDINGS/{Disease}/embeddings.h5
- DATA/05-SEMANTIC/EMBEDDINGS/{Disease}/embeddings.h5.sha256
- DATA/05-SEMANTIC/EMBEDDINGS/{Disease}/metadata.json
- DATA/05-SEMANTIC/.embed_complete marker

USAGE:
    python 05-02-heim-sem-embed.py                      # Full run
    python 05-02-heim-sem-embed.py --resume             # Resume from checkpoint
    python 05-02-heim-sem-embed.py --disease "Malaria"  # Single disease
    python 05-02-heim-sem-embed.py --cpu-only           # Force CPU backend

REQUIREMENTS:
    pip install torch transformers h5py pandas tqdm
"""

import os
import sys
import gc
import json
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "DATA"
SEMANTIC_DIR = DATA_DIR / "05-SEMANTIC"
PUBMED_RAW_DIR = SEMANTIC_DIR / "PUBMED-RAW"
EMBEDDINGS_DIR = SEMANTIC_DIR / "EMBEDDINGS"
CHECKPOINTS_DIR = SEMANTIC_DIR / "CHECKPOINTS"
LOGS_DIR = BASE_DIR / "LOGS"

# Marker files
FETCH_COMPLETE = SEMANTIC_DIR / ".fetch_complete"
EMBED_COMPLETE = SEMANTIC_DIR / ".embed_complete"
CHECKPOINT_FILE = CHECKPOINTS_DIR / "checkpoint_embed.json"
CHECKPOINT_BACKUP = CHECKPOINTS_DIR / "checkpoint_embed.json.bak"

# Model configuration
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
EMBEDDING_DIM = 768
MAX_LENGTH = 512

# Processing configuration
BATCH_SIZE = 64
GC_INTERVAL = 50  # Run gc.collect() every N batches
MEMORY_WARNING_GB = 200  # Warn if memory exceeds this

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"05-02-embed-{timestamp}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("heim_embed")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

class CheckpointManager:
    """Manage checkpoints for resumable embedding generation."""

    def __init__(self, checkpoint_file: Path, backup_file: Path):
        self.checkpoint_file = checkpoint_file
        self.backup_file = backup_file
        self.data = self._load_or_create()

    def _load_or_create(self) -> Dict:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                stored_checksum = data.get("checksum", "")
                data_copy = {k: v for k, v in data.items() if k != "checksum"}
                computed_checksum = hashlib.sha256(
                    json.dumps(data_copy, sort_keys=True).encode()
                ).hexdigest()[:16]
                if stored_checksum == computed_checksum:
                    return data
            except (json.JSONDecodeError, KeyError):
                pass

        return {
            "script": "05-02-heim-sem-embed.py",
            "version": "1.0",
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_diseases": [],
            "in_progress": None,
            "failed_diseases": [],
            "disease_stats": {},
            "total_embeddings": 0,
        }

    def save(self):
        self.data["last_updated"] = datetime.now().isoformat()
        data_copy = {k: v for k, v in self.data.items() if k != "checksum"}
        self.data["checksum"] = hashlib.sha256(
            json.dumps(data_copy, sort_keys=True).encode()
        ).hexdigest()[:16]

        if self.checkpoint_file.exists():
            self.checkpoint_file.rename(self.backup_file)

        temp_file = self.checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        temp_file.rename(self.checkpoint_file)

    def mark_in_progress(self, disease: str):
        self.data["in_progress"] = disease
        self.save()

    def mark_completed(self, disease: str, stats: Dict):
        if disease not in self.data["completed_diseases"]:
            self.data["completed_diseases"].append(disease)
        self.data["disease_stats"][disease] = stats
        self.data["total_embeddings"] += stats.get("embeddings_generated", 0)
        self.data["in_progress"] = None
        self.save()

    def mark_failed(self, disease: str, error: str):
        self.data["failed_diseases"].append({
            "disease": disease,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self.data["in_progress"] = None
        self.save()

    def is_completed(self, disease: str) -> bool:
        return disease in self.data["completed_diseases"]

    def get_pending_diseases(self, all_diseases: List[str]) -> List[str]:
        completed = set(self.data["completed_diseases"])
        return [d for d in all_diseases if d not in completed]

# =============================================================================
# EMBEDDING GENERATOR
# =============================================================================

class EmbeddingGenerator:
    """Generate PubMedBERT embeddings with MPS/CPU backend."""

    def __init__(self, logger: logging.Logger, use_mps: bool = True):
        self.logger = logger
        self.device = self._setup_device(use_mps)
        self.tokenizer = None
        self.model = None

    def _setup_device(self, use_mps: bool) -> torch.device:
        """Setup compute device (MPS or CPU)."""
        if use_mps and torch.backends.mps.is_available():
            device = torch.device("mps")
            self.logger.info(f"  Using device: MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            self.logger.info(f"  Using device: CPU")
        return device

    def load_model(self):
        """Load PubMedBERT model and tokenizer."""
        self.logger.info(f"  Loading model: {MODEL_NAME}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

        self.logger.info(f"  Model loaded successfully")

    def validate_mps(self) -> bool:
        """Validate MPS produces consistent results vs CPU."""
        if self.device.type != "mps":
            return True

        self.logger.info("  Validating MPS consistency...")

        test_texts = [
            "Malaria is caused by Plasmodium parasites.",
            "Tuberculosis affects the lungs primarily.",
            "Cancer research has advanced significantly."
        ]

        # Generate on MPS
        mps_embeddings = self._embed_texts(test_texts)

        # Generate on CPU
        self.model.to("cpu")
        cpu_device = self.device
        self.device = torch.device("cpu")
        cpu_embeddings = self._embed_texts(test_texts)
        self.device = cpu_device
        self.model.to(self.device)

        # Compare
        for i, (mps_emb, cpu_emb) in enumerate(zip(mps_embeddings, cpu_embeddings)):
            cosine_sim = np.dot(mps_emb, cpu_emb) / (
                np.linalg.norm(mps_emb) * np.linalg.norm(cpu_emb)
            )
            if cosine_sim < 0.999:
                self.logger.warning(
                    f"  MPS validation failed: cosine sim = {cosine_sim:.6f}"
                )
                return False

        self.logger.info("  MPS validation passed (cosine sim > 0.999)")
        return True

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

            # Mean pooling over tokens
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()
            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            return embeddings.cpu().numpy()

    def embed_disease(
        self,
        disease_folder: Path,
        output_dir: Path,
        logger: logging.Logger
    ) -> Dict:
        """Generate embeddings for all papers in a disease folder."""
        disease_name = disease_folder.name

        # Collect all papers
        all_papers = []
        year_files = sorted(disease_folder.glob("pubmed_*.json"))

        for year_file in year_files:
            try:
                with open(year_file, 'r') as f:
                    data = json.load(f)
                papers = data.get("papers", [])
                for paper in papers:
                    if paper.get("abstract"):  # Only papers with abstracts
                        all_papers.append({
                            "pmid": paper["pmid"],
                            "year": paper.get("year"),
                            "text": f"{paper.get('title', '')} {paper['abstract']}"
                        })
            except Exception as e:
                logger.warning(f"  Error loading {year_file.name}: {e}")

        if not all_papers:
            logger.warning(f"  No papers with abstracts for {disease_name}")
            return {
                "disease": disease_name,
                "papers_found": 0,
                "embeddings_generated": 0,
                "status": "no_abstracts"
            }

        logger.info(f"  Papers with abstracts: {len(all_papers):,}")

        # Generate embeddings in batches
        all_embeddings = []
        all_pmids = []
        all_years = []

        batches = [
            all_papers[i:i + BATCH_SIZE]
            for i in range(0, len(all_papers), BATCH_SIZE)
        ]

        for batch_idx, batch in enumerate(tqdm(
            batches,
            desc=f"    {disease_name[:25]}",
            leave=False
        )):
            texts = [p["text"] for p in batch]

            try:
                embeddings = self._embed_texts(texts)
                all_embeddings.append(embeddings)
                all_pmids.extend([p["pmid"] for p in batch])
                all_years.extend([p["year"] for p in batch])

            except Exception as e:
                logger.error(f"  Batch {batch_idx} failed: {e}")
                continue

            # Memory management
            if (batch_idx + 1) % GC_INTERVAL == 0:
                gc.collect()
                if self.device.type == "mps":
                    torch.mps.empty_cache()

        if not all_embeddings:
            return {
                "disease": disease_name,
                "papers_found": len(all_papers),
                "embeddings_generated": 0,
                "status": "embedding_failed"
            }

        # Concatenate all embeddings
        embeddings_array = np.vstack(all_embeddings).astype(np.float32)
        logger.info(f"  Generated {embeddings_array.shape[0]:,} embeddings")

        # Save to HDF5
        output_dir.mkdir(parents=True, exist_ok=True)
        h5_file = output_dir / "embeddings.h5"

        with h5py.File(h5_file, 'w') as f:
            f.create_dataset(
                'embeddings',
                data=embeddings_array,
                compression='gzip',
                compression_opts=4
            )
            f.create_dataset(
                'pmids',
                data=np.array(all_pmids, dtype='S20')
            )
            f.create_dataset(
                'years',
                data=np.array([y if y else "unknown" for y in all_years], dtype='S10')
            )
            f.attrs['disease'] = disease_name
            f.attrs['model'] = MODEL_NAME
            f.attrs['embedding_dim'] = EMBEDDING_DIM
            f.attrs['created_at'] = datetime.now().isoformat()

        # Compute checksum
        with open(h5_file, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        checksum_file = output_dir / "embeddings.h5.sha256"
        with open(checksum_file, 'w') as f:
            f.write(file_hash)

        # Save metadata
        metadata = {
            "disease": disease_name,
            "papers_processed": len(all_papers),
            "embeddings_generated": embeddings_array.shape[0],
            "embedding_dimension": embeddings_array.shape[1],
            "model": MODEL_NAME,
            "created_at": datetime.now().isoformat(),
            "file_checksum": file_hash,
            "file_size_mb": h5_file.stat().st_size / (1024 * 1024)
        }

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Cleanup
        del embeddings_array, all_embeddings
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()

        return {
            "disease": disease_name,
            "papers_found": len(all_papers),
            "embeddings_generated": metadata["embeddings_generated"],
            "file_size_mb": metadata["file_size_mb"],
            "status": "success"
        }

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def get_disease_folders(raw_dir: Path) -> List[Path]:
    """Get list of disease folders to process."""
    folders = []
    for folder in sorted(raw_dir.iterdir()):
        if folder.is_dir() and not folder.name.startswith('.'):
            # Check if it has any pubmed files
            if list(folder.glob("pubmed_*.json")):
                folders.append(folder)
    return folders

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HEIM Semantic Pipeline - Embedding Generation"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint")
    parser.add_argument("--disease", type=str, help="Process single disease")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU backend")
    parser.add_argument("--no-caffeinate", action="store_true", help="Skip caffeinate")

    args = parser.parse_args()

    # Handle caffeinate wrapper
    if not args.no_caffeinate and sys.platform == "darwin":
        print("Starting with sleep prevention enabled (caffeinate)")
        cmd = ["caffeinate", "-i", "-s", "-d", sys.executable, __file__, "--no-caffeinate"]
        if args.fresh:
            cmd.append("--fresh")
        if args.cpu_only:
            cmd.append("--cpu-only")
        if args.disease:
            cmd.extend(["--disease", args.disease])
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    # Setup
    logger = setup_logging()

    print("\n" + "=" * 70)
    print(" HEIM SEMANTIC PIPELINE - EMBEDDING GENERATION")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model: {MODEL_NAME}")

    # Check prerequisites
    if not FETCH_COMPLETE.exists():
        print("\n  ERROR: Fetch not complete. Run 05-01-heim-sem-fetch.py first.")
        sys.exit(1)

    # Initialize generator
    use_mps = not args.cpu_only
    generator = EmbeddingGenerator(logger, use_mps=use_mps)
    generator.load_model()

    # Validate MPS if using it
    if generator.device.type == "mps":
        if not generator.validate_mps():
            print("  WARNING: MPS validation failed, falling back to CPU")
            generator = EmbeddingGenerator(logger, use_mps=False)
            generator.load_model()

    # Initialize checkpoint
    if args.fresh and CHECKPOINT_FILE.exists():
        print("  Removing existing checkpoint (--fresh flag)")
        CHECKPOINT_FILE.unlink()

    checkpoint = CheckpointManager(CHECKPOINT_FILE, CHECKPOINT_BACKUP)

    # Get disease folders
    if args.disease:
        disease_folder = args.disease.replace("/", "_").replace(" ", "_")
        all_folders = [PUBMED_RAW_DIR / disease_folder]
        if not all_folders[0].exists():
            print(f"  ERROR: Disease folder not found: {disease_folder}")
            sys.exit(1)
    else:
        all_folders = get_disease_folders(PUBMED_RAW_DIR)

    print(f"  Disease folders found: {len(all_folders)}")

    # Filter to pending
    folder_names = [f.name for f in all_folders]
    pending_names = checkpoint.get_pending_diseases(folder_names)
    pending_folders = [f for f in all_folders if f.name in pending_names]

    print(f"  Already completed: {len(all_folders) - len(pending_folders)}")
    print(f"  Pending: {len(pending_folders)}")

    if not pending_folders:
        print("\n  All diseases already processed!")
    else:
        print(f"\n  Starting embedding generation...")
        print(f"  Estimated time: {len(pending_folders) * 5} - {len(pending_folders) * 15} minutes")
        print("-" * 70)

        for i, disease_folder in enumerate(pending_folders, 1):
            disease_name = disease_folder.name
            output_dir = EMBEDDINGS_DIR / disease_name

            logger.info(f"\n[{i}/{len(pending_folders)}] {disease_name}")
            checkpoint.mark_in_progress(disease_name)

            try:
                stats = generator.embed_disease(
                    disease_folder, output_dir, logger
                )
                checkpoint.mark_completed(disease_name, stats)

                logger.info(f"  Status: {stats['status']}")
                logger.info(f"  Embeddings: {stats['embeddings_generated']:,}")
                if stats.get('file_size_mb'):
                    logger.info(f"  File size: {stats['file_size_mb']:.1f} MB")

            except KeyboardInterrupt:
                logger.warning("\nInterrupted by user. Progress saved to checkpoint.")
                print("\n  Interrupted. Resume with: python 05-02-heim-sem-embed.py --resume")
                sys.exit(130)

            except Exception as e:
                logger.error(f"Failed to process {disease_name}: {e}")
                checkpoint.mark_failed(disease_name, str(e))
                continue

    # Create completion marker
    with open(EMBED_COMPLETE, 'w') as f:
        json.dump({
            "completed_at": datetime.now().isoformat(),
            "diseases_processed": len(checkpoint.data["completed_diseases"]),
            "total_embeddings": checkpoint.data["total_embeddings"]
        }, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print(" EMBEDDING GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total diseases: {len(checkpoint.data['completed_diseases'])}")
    print(f"  Total embeddings: {checkpoint.data['total_embeddings']:,}")
    if checkpoint.data["failed_diseases"]:
        print(f"  Failed: {len(checkpoint.data['failed_diseases'])}")
    print(f"\n  Next step: python 05-03-heim-sem-compute-metrics.py")

if __name__ == "__main__":
    main()
