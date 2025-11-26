#!/usr/bin/env python3
"""Train teacher models (CatBoost classifier + LightGBM regressor).

Usage:
    # Fast iteration (Oct 1 only, ~3M rows)
    uv run python scripts/train_teachers.py --subset

    # Full training (Oct 1-5, ~17M rows)
    uv run python scripts/train_teachers.py --full

This script:
1. Loads training data with time-based train/val split
2. Generates lookup tables from training data
3. Builds offline features (rich features for teachers)
4. Trains CatBoost buyer classifier
5. Trains LightGBM revenue regressor with HistOS sampling
6. Saves models and soft labels for distillation
"""

from __future__ import annotations

import argparse
import gc
import sys
from copy import deepcopy
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import catboost as cb
import dask.dataframe as dd
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src.data.loader import DataLoader
from src.features.lookup_tables import generate_lookup_tables
from src.features.offline import build_offline_features
from src.models.histos_sampling import histos_sample
from src.utils.logger import get_logger
from src.utils.metrics import evaluate_classifier, evaluate_regressor, print_metrics

LOGGER = get_logger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "teachers"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def apply_two_stage_prediction(
    buyer_proba: np.ndarray,
    log_revenue_pred: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """Apply two-stage prediction: zero revenue for predicted non-buyers.
    
    Args:
        buyer_proba: Buyer probability predictions from classifier
        log_revenue_pred: Log-revenue predictions from regressor
        threshold: Classification threshold (default 0.5)
        
    Returns:
        Revenue predictions with zeros for non-buyers
        
    Postconditions:
        - Non-buyers (proba < threshold) always get exactly 0.0 revenue
        - All predictions are non-negative
    """
    # Determine buyers
    is_buyer = buyer_proba >= threshold
    
    # Initialize all predictions to zero
    revenue_pred = np.zeros(len(buyer_proba), dtype=np.float64)
    
    # Only predict revenue for buyers
    if is_buyer.sum() > 0:
        revenue_pred[is_buyer] = np.expm1(log_revenue_pred[is_buyer])
        revenue_pred[is_buyer] = np.maximum(revenue_pred[is_buyer], 0.0)
    
    # Postcondition: verify non-buyers have zero revenue
    assert np.all(revenue_pred[~is_buyer] == 0.0), "Non-buyers must have zero revenue"
    assert np.all(revenue_pred >= 0.0), "All predictions must be non-negative"
    
    return revenue_pred


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train teacher models")
    parser.add_argument(
        "--mode",
        choices=["subset", "full"],
        default="subset",
        help="Training mode: 'subset' (1 day, fast) or 'full' (5 days, slow)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of training days (1-7). Overrides mode. Default: 1 for subset, 5 for full"
    )
    parser.add_argument(
        "--batch-days",
        type=int,
        default=None,
        help="Load data in batches of N days for progressive caching. Default: load all at once"
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate on actual test data (Oct 8-11) to measure true MSLE"
    )
    parser.add_argument(
        "--eval-fallback-start",
        type=str,
        default=None,
        help="Fallback start datetime (YYYY-MM-DD-HH-MM) when test set lacks targets (defaults to val_start)",
    )
    parser.add_argument(
        "--eval-fallback-end",
        type=str,
        default=None,
        help="Fallback end datetime (YYYY-MM-DD-HH-MM) when test set lacks targets (defaults to val_end)",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Invalidate and rebuild feature cache (useful after code changes)"
    )
    # Backward compatibility
    parser.add_argument("--subset", action="store_const", const="subset", dest="mode")
    parser.add_argument("--full", action="store_const", const="full", dest="mode")
    
    return parser.parse_args()


def generate_date_batches(start_date: str, end_date: str, batch_days: int) -> list[tuple[str, str]]:
    """Generate batches of dates for progressive loading.
    
    Args:
        start_date: Start date in format "YYYY-MM-DD-HH-MM"
        end_date: End date in format "YYYY-MM-DD-HH-MM"
        batch_days: Number of days per batch
        
    Returns:
        List of (batch_start, batch_end) tuples
        
    Example:
        >>> generate_date_batches("2025-10-01-00-00", "2025-10-03-23-00", 1)
        [("2025-10-01-00-00", "2025-10-01-23-00"),
         ("2025-10-02-00-00", "2025-10-02-23-00"),
         ("2025-10-03-00-00", "2025-10-03-23-00")]
    """
    from datetime import datetime, timedelta
    
    # Parse dates
    fmt = "%Y-%m-%d-%H-%M"
    start = datetime.strptime(start_date, fmt)
    end = datetime.strptime(end_date, fmt)
    
    batches = []
    current = start
    
    while current <= end:
        batch_end = min(
            current + timedelta(days=batch_days) - timedelta(hours=1),
            end
        )
        batches.append((
            current.strftime(fmt),
            batch_end.strftime(fmt)
        ))
        current = batch_end + timedelta(hours=1)
    
    return batches


def load_labeled_eval_data(
    base_config: dict,
    fallback_start: str,
    fallback_end: str,
) -> tuple[dd.DataFrame, str]:
    """Load a labeled evaluation slice from the training parquet.

    Args:
        base_config: Original configuration dictionary
        fallback_start: Start datetime (inclusive)
        fallback_end: End datetime (inclusive)

    Returns:
        Tuple of (evaluation dataframe, description of slice)
    """
    eval_config = deepcopy(base_config)
    eval_config["data"] = dict(base_config["data"])
    eval_config["data"]["test_path"] = eval_config["data"]["train_path"]
    eval_config["data"]["test_start"] = fallback_start
    eval_config["data"]["test_end"] = fallback_end

    eval_loader = DataLoader(eval_config)
    ddf_eval = eval_loader.load_test()
    slice_desc = f"{fallback_start} -> {fallback_end} (train parquet)"
    return ddf_eval, slice_desc


def load_and_cache_batch(
    batch_start: str,
    batch_end: str,
    val_start: str,
    val_end: str,
    config: dict,
    lookup_tables: dict,
    cache_dir: Path,
    cache_key_base: str,
    batch_idx: int,
    total_batches: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load and cache features for a single batch.
    
    Returns:
        Tuple of (X_train_batch, X_val_batch, targets_dict)
    """
    import pickle
    
    # Create batch-specific cache key
    batch_cache_key = f"{cache_key_base}_batch_{batch_start}_{batch_end}"
    batch_cache_train_path = cache_dir / f"{batch_cache_key}_train.parquet"
    batch_cache_val_path = cache_dir / f"{batch_cache_key}_val.parquet"
    batch_cache_targets_path = cache_dir / f"{batch_cache_key}_targets.pkl"
    
    # Check if this batch is cached
    if (batch_cache_train_path.exists() and 
        batch_cache_val_path.exists() and 
        batch_cache_targets_path.exists()):
        LOGGER.info(f"  Batch {batch_idx}/{total_batches} [{batch_start} to {batch_end}]: ✅ Found cached features")
        X_train_batch = pd.read_parquet(batch_cache_train_path)
        X_val_batch = pd.read_parquet(batch_cache_val_path)
        with open(batch_cache_targets_path, "rb") as f:
            targets_batch = pickle.load(f)
        return X_train_batch, X_val_batch, targets_batch
    
    # Not cached - need to compute
    LOGGER.info(f"  Batch {batch_idx}/{total_batches} [{batch_start} to {batch_end}]: Computing features...")
    
    # Create config for this batch
    batch_config = config.copy()
    batch_config["data"]["train_start"] = batch_start
    batch_config["data"]["train_end"] = batch_end
    
    # Load data for this batch
    batch_loader = DataLoader(batch_config)
    ddf_train_batch, ddf_val_batch = batch_loader.load_train(validation_split=True)
    
    # Build features
    LOGGER.info(f"  Batch {batch_idx}: Building training features...")
    X_train_dask_batch, encoders_batch = build_offline_features(
        ddf_train_batch,
        lookup_tables=lookup_tables
    )
    LOGGER.info(f"  Batch {batch_idx}: ✅ Training features built (lazy, {X_train_dask_batch.npartitions} partitions)")
    
    LOGGER.info(f"  Batch {batch_idx}: Building validation features...")
    X_val_dask_batch, _ = build_offline_features(
        ddf_val_batch,
        lookup_tables=lookup_tables,
        encoders_from_online=encoders_batch
    )
    LOGGER.info(f"  Batch {batch_idx}: ✅ Validation features built (lazy, {X_val_dask_batch.npartitions} partitions)")
    
    # Compute features with progress logging
    import time
    from dask.callbacks import Callback
    
    class LoggingCallback(Callback):
        def __init__(self, name: str, total: int):
            self.name = name
            self.total = total
            self.completed = 0
            self.start_time = None
            
        def _start(self, dsk):
            self.start_time = time.time()
            LOGGER.info(f"  [{self.name}] Starting computation of {self.total} tasks...")
            
        def _pretask(self, key, dsk, state):
            self.completed += 1
            # Report every 100 tasks or at milestones (10%, 25%, 50%, 75%, 90%, 100%)
            milestones = [int(self.total * p) for p in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]]
            if self.completed % 100 == 0 or self.completed in milestones:
                elapsed = time.time() - self.start_time
                rate = self.completed / elapsed if elapsed > 0 else 0
                remaining = (self.total - self.completed) / rate if rate > 0 else 0
                LOGGER.info(
                    f"  [{self.name}] Progress: {self.completed}/{self.total} tasks "
                    f"({100*self.completed/self.total:.1f}%) | "
                    f"Rate: {rate:.1f} tasks/sec | "
                    f"ETA: {remaining:.0f}s"
                )
        
        def _finish(self, dsk, state, errored):
            elapsed = time.time() - self.start_time
            LOGGER.info(f"  [{self.name}] Completed in {elapsed:.1f}s")
    
    LOGGER.info(f"  Batch {batch_idx}: Computing training features ({X_train_dask_batch.npartitions} partitions)...")
    import sys
    sys.stdout.flush()
    total_train_tasks = X_train_dask_batch.npartitions * 175
    LOGGER.info(f"  Batch {batch_idx}: Estimated {total_train_tasks} tasks for training features")
    sys.stdout.flush()
    
    callback_train = LoggingCallback("BATCH_TRAIN", total_train_tasks)
    with callback_train:
        LOGGER.info(f"  Batch {batch_idx}: Starting compute() for training features...")
        sys.stdout.flush()
        X_train_batch = X_train_dask_batch.compute()
    LOGGER.info(f"  Batch {batch_idx}: ✅ Training features computed: {X_train_batch.shape}")
    sys.stdout.flush()
    
    LOGGER.info(f"  Batch {batch_idx}: Computing validation features ({X_val_dask_batch.npartitions} partitions)...")
    sys.stdout.flush()
    total_val_tasks = X_val_dask_batch.npartitions * 175
    LOGGER.info(f"  Batch {batch_idx}: Estimated {total_val_tasks} tasks for validation features")
    sys.stdout.flush()
    
    callback_val = LoggingCallback("BATCH_VAL", total_val_tasks)
    with callback_val:
        LOGGER.info(f"  Batch {batch_idx}: Starting compute() for validation features...")
        sys.stdout.flush()
        X_val_batch = X_val_dask_batch.compute()
    LOGGER.info(f"  Batch {batch_idx}: ✅ Validation features computed: {X_val_batch.shape}")
    sys.stdout.flush()
    
    # Filter to numeric columns
    LOGGER.info(f"  Filtering to numeric columns for batch {batch_idx}...")
    numeric_cols = X_train_batch.select_dtypes(include=[np.number]).columns.tolist()
    X_train_batch = X_train_batch[numeric_cols]
    X_val_batch = X_val_batch[numeric_cols]
    
    # Get targets
    LOGGER.info(f"  Extracting targets for batch {batch_idx}...")
    y_cls_train_batch = ddf_train_batch["buyer_d7"].compute().values
    y_cls_val_batch = ddf_val_batch["buyer_d7"].compute().values
    y_reg_train_batch = ddf_train_batch["iap_revenue_d7"].compute().values
    y_reg_val_batch = ddf_val_batch["iap_revenue_d7"].compute().values
    
    targets_batch = {
        "y_cls_train": y_cls_train_batch,
        "y_cls_val": y_cls_val_batch,
        "y_reg_train": y_reg_train_batch,
        "y_reg_val": y_reg_val_batch
    }
    
    # Cache this batch
    X_train_batch.to_parquet(batch_cache_train_path, compression="snappy")
    X_val_batch.to_parquet(batch_cache_val_path, compression="snappy")
    with open(batch_cache_targets_path, "wb") as f:
        pickle.dump(targets_batch, f)
    
    LOGGER.info(f"  Batch {batch_idx}/{total_batches}: ✅ Computed and cached")
    
    return X_train_batch, X_val_batch, targets_batch


def main():
    """Main training pipeline."""
    args = parse_args()
    
    LOGGER.info("=" * 80)
    LOGGER.info("TEACHER TRAINING PIPELINE")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Mode: {args.mode}")
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 1: Loading Data")
    LOGGER.info("=" * 80)
    
    # Determine number of training days
    if args.days is not None:
        num_days = args.days
        assert 1 <= num_days <= 7, "Number of days must be between 1 and 7"
    else:
        # Default based on mode
        num_days = 1 if args.mode == "subset" else 5
    
    LOGGER.info(f"Training on {num_days} day(s) of data")
    
    # Time-based split configuration
    train_start = "2025-10-01-00-00"
    
    # Calculate end date based on number of days
    if num_days == 1:
        train_end = "2025-10-01-23-00"
    elif num_days == 2:
        train_end = "2025-10-02-23-00"
    elif num_days == 3:
        train_end = "2025-10-03-23-00"
    elif num_days == 4:
        train_end = "2025-10-04-23-00"
    elif num_days == 5:
        train_end = "2025-10-05-23-00"
    elif num_days == 6:
        train_end = "2025-10-06-23-00"
    else:  # 7 days
        train_end = "2025-10-07-23-00"
    
    LOGGER.info(f"Train period: {train_start} to {train_end}")
    
    # Validation: Oct 6 (always the same)
    val_start = "2025-10-06-00-00"
    val_end = "2025-10-06-23-00"
    
    config = {
        "data": {
            "train_path": str(DATA_DIR / "raw" / "train" / "train"),
            "test_path": str(DATA_DIR / "raw" / "test" / "test"),
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": "2025-10-07-00-00",  # Placeholder
            "test_end": "2025-10-07-23-00"     # Placeholder
        },
        "dask": {
            "client": {
                "enabled": False,  # Disabled to avoid memory issues with large dataset
                "n_workers": 4,
                "threads_per_worker": 4
            }
        }
    }
    
    loader = DataLoader(config)
    ddf_train, ddf_val = loader.load_train(validation_split=True)
    
    LOGGER.info(f"Train partitions: {ddf_train.npartitions}")
    LOGGER.info(f"Val partitions: {ddf_val.npartitions}")
    
    # =========================================================================
    # 2. Generate Lookup Tables
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 2: Generating Lookup Tables")
    LOGGER.info("=" * 80)
    
    lookup_dir = PROCESSED_DIR / "lookup_tables"
    lookup_dir.mkdir(parents=True, exist_ok=True)
    
    lookup_tables = generate_lookup_tables(
        ddf_train,
        output_dir=lookup_dir
    )
    
    # =========================================================================
    # 3. Build Offline Features
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 3: Building Offline Features")
    LOGGER.info("=" * 80)
    
    import pickle
    cache_dir = Path("data/processed/features_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key_base = f"train_{train_start}_{train_end}_val_{val_start}_{val_end}_v2"
    
    # Handle cache invalidation if requested
    if args.rebuild_cache:
        LOGGER.info("🗑️  --rebuild-cache flag set: Invalidating existing cache...")
        cache_pattern = f"{cache_key_base}*.parquet"
        cache_files = list(cache_dir.glob(cache_pattern))
        targets_pattern = f"{cache_key_base}*.pkl"
        cache_files.extend(list(cache_dir.glob(targets_pattern)))
        
        if cache_files:
            LOGGER.info(f"   Found {len(cache_files)} cache file(s) to delete")
            for cache_file in cache_files:
                cache_file.unlink()
                LOGGER.info(f"   Deleted: {cache_file.name}")
            LOGGER.info("✅ Cache invalidated successfully")
        else:
            LOGGER.info("   No cache files found to delete")
    
    # Check if using batched loading
    if args.batch_days is not None:
        LOGGER.info(f"Using batched loading with {args.batch_days} day(s) per batch...")
        
        # Generate date batches
        date_batches = generate_date_batches(train_start, train_end, args.batch_days)
        LOGGER.info(f"Split into {len(date_batches)} batch(es)")
        
        # Load and cache each batch
        X_train_batches = []
        X_val_batches = []
        y_cls_train_batches = []
        y_cls_val_batches = []
        y_reg_train_batches = []
        y_reg_val_batches = []
        encoders = None
        
        for batch_idx, (batch_start, batch_end) in enumerate(date_batches, 1):
            X_train_batch, X_val_batch, targets_batch = load_and_cache_batch(
                batch_start, batch_end, val_start, val_end,
                config, lookup_tables, cache_dir, cache_key_base,
                batch_idx, len(date_batches)
            )
            
            X_train_batches.append(X_train_batch)
            X_val_batches.append(X_val_batch)
            y_cls_train_batches.append(targets_batch["y_cls_train"])
            y_cls_val_batches.append(targets_batch["y_cls_val"])
            y_reg_train_batches.append(targets_batch["y_reg_train"])
            y_reg_val_batches.append(targets_batch["y_reg_val"])
            
            # Save encoders from first batch (they should be the same)
            if encoders is None:
                # Load encoders from first batch (they're saved during feature building)
                encoder_path = PROCESSED_DIR / "encoders" / "online_encoders.pkl"
                if encoder_path.exists():
                    with open(encoder_path, "rb") as f:
                        encoders = pickle.load(f)
        
        # Concatenate all batches
        LOGGER.info("Concatenating batches...")
        X_train = pd.concat(X_train_batches, ignore_index=True)
        X_val = pd.concat(X_val_batches, ignore_index=True)
        y_cls_train = np.concatenate(y_cls_train_batches)
        y_cls_val = np.concatenate(y_cls_val_batches)
        y_reg_train = np.concatenate(y_reg_train_batches)
        y_reg_val = np.concatenate(y_reg_val_batches)
        
        LOGGER.info(f"✅ Combined {len(date_batches)} batch(es): Train {X_train.shape}, Val {X_val.shape}")
        
    else:
        # Original non-batched logic
        LOGGER.info("Building features for training data...")
        X_train_dask, encoders = build_offline_features(
            ddf_train,
            lookup_tables=lookup_tables
        )
        
        LOGGER.info("Building features for validation data...")
        X_val_dask, _ = build_offline_features(
            ddf_val,
            lookup_tables=lookup_tables,
            encoders_from_online=encoders
        )
        
        # Save encoders
        encoder_path = PROCESSED_DIR / "encoders" / "online_encoders.pkl"
        encoder_path.parent.mkdir(parents=True, exist_ok=True)
        with open(encoder_path, "wb") as f:
            pickle.dump(encoders, f)
        LOGGER.info(f"✅ Saved encoders to {encoder_path}")
        
        # Check for cached features
        cache_key = cache_key_base
        cache_train_path = cache_dir / f"{cache_key}_train.parquet"
        cache_val_path = cache_dir / f"{cache_key}_val.parquet"
        cache_targets_path = cache_dir / f"{cache_key}_targets.pkl"
        
        if cache_train_path.exists() and cache_val_path.exists() and cache_targets_path.exists():
            LOGGER.info("✅ Found cached features! Loading from disk...")
            LOGGER.info(f"  Train cache: {cache_train_path}")
            LOGGER.info(f"  Val cache: {cache_val_path}")
            
            X_train = pd.read_parquet(cache_train_path)
            X_val = pd.read_parquet(cache_val_path)
            
            with open(cache_targets_path, "rb") as f:
                targets = pickle.load(f)
            y_cls_train = targets["y_cls_train"]
            y_cls_val = targets["y_cls_val"]
            y_reg_train = targets["y_reg_train"]
            y_reg_val = targets["y_reg_val"]
            
            LOGGER.info(f"✅ Loaded from cache in <1s (saved ~5 minutes!)")
            LOGGER.info(f"  Train: {X_train.shape}")
            LOGGER.info(f"  Val: {X_val.shape}")
        else:
            LOGGER.info("No cache found. Computing features (this will take ~5 minutes)...")
            LOGGER.info(f"Training: {X_train_dask.npartitions} partitions to process")
            LOGGER.info(f"Validation: {X_val_dask.npartitions} partitions to process")
            
            # Custom progress callback for better logging
            import time
            from dask.callbacks import Callback
            
            class LoggingCallback(Callback):
                def __init__(self, name: str, total: int):
                    self.name = name
                    self.total = total
                    self.completed = 0
                    self.start_time = None
                    
                def _start(self, dsk):
                    self.start_time = time.time()
                    LOGGER.info(f"[{self.name}] Starting computation of {self.total} tasks...")
                    
                def _pretask(self, key, dsk, state):
                    self.completed += 1
                    # Report every 100 tasks or at milestones (10%, 25%, 50%, 75%, 90%, 100%)
                    milestones = [int(self.total * p) for p in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]]
                    if self.completed % 100 == 0 or self.completed in milestones:
                        elapsed = time.time() - self.start_time
                        rate = self.completed / elapsed if elapsed > 0 else 0
                        remaining = (self.total - self.completed) / rate if rate > 0 else 0
                        LOGGER.info(
                            f"[{self.name}] Progress: {self.completed}/{self.total} tasks "
                            f"({100*self.completed/self.total:.1f}%) | "
                            f"Rate: {rate:.1f} tasks/sec | "
                            f"ETA: {remaining:.0f}s"
                        )
                
                def _finish(self, dsk, state, errored):
                    elapsed = time.time() - self.start_time
                    LOGGER.info(f"[{self.name}] Completed in {elapsed:.1f}s")
            
            LOGGER.info("Progress: Computing training features...")
            # Realistic task estimate: ~175 tasks per partition (observed: 4200 tasks / 24 partitions)
            with LoggingCallback("TRAIN", X_train_dask.npartitions * 175):
                X_train = X_train_dask.compute()
            LOGGER.info(f"✅ Training features computed: {X_train.shape}")
            
            LOGGER.info("Progress: Computing validation features...")
            # Same multiplier for validation
            with LoggingCallback("VAL", X_val_dask.npartitions * 175):
                X_val = X_val_dask.compute()
            LOGGER.info(f"✅ Validation features computed: {X_val.shape}")
            
            # Filter to only numeric columns immediately after compute (before caching)
            LOGGER.info("Filtering to numeric columns...")
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < len(X_train.columns):
                dropped = set(X_train.columns) - set(numeric_cols)
                LOGGER.warning(f"Dropping {len(dropped)} non-numeric columns: {list(dropped)[:10]}")
            X_train = X_train[numeric_cols]
            X_val = X_val[numeric_cols]
            LOGGER.info(f"✅ Filtered to {len(numeric_cols)} numeric columns")
            
            # Get targets
            LOGGER.info("Progress: Extracting targets...")
            y_cls_train = ddf_train["buyer_d7"].compute().values
            y_cls_val = ddf_val["buyer_d7"].compute().values
            
            y_reg_train = ddf_train["iap_revenue_d7"].compute().values
            y_reg_val = ddf_val["iap_revenue_d7"].compute().values
            LOGGER.info("✅ Targets extracted")
            
            # Cache the computed features
            LOGGER.info("Caching features for future runs...")
            X_train.to_parquet(cache_train_path, compression="snappy")
            X_val.to_parquet(cache_val_path, compression="snappy")
            
            targets = {
                "y_cls_train": y_cls_train,
                "y_cls_val": y_cls_val,
                "y_reg_train": y_reg_train,
                "y_reg_val": y_reg_val
            }
            with open(cache_targets_path, "wb") as f:
                pickle.dump(targets, f)
            
            LOGGER.info(f"✅ Features cached to {cache_dir}")
    
    # Features are already filtered to numeric columns
    LOGGER.info(f"✅ Train: {X_train.shape}, {X_train.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    LOGGER.info(f"✅ Val: {X_val.shape}, {X_val.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    # Convert to log scale for regressor
    log_rev_train = np.log1p(y_reg_train)
    log_rev_val = np.log1p(y_reg_val)
    
    LOGGER.info(f"Buyer rate (train): {y_cls_train.mean():.4f}")
    LOGGER.info(f"Buyer rate (val): {y_cls_val.mean():.4f}")
    
    # =========================================================================
    # 4. Train Teacher Classifier (CatBoost)
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 4: Training CatBoost Buyer Classifier")
    LOGGER.info("=" * 80)
    
    model_cls_path = MODELS_DIR / "teacher_classifier_catboost.cbm"
    
    # Check if model already exists
    if model_cls_path.exists():
        LOGGER.info(f"✅ Found cached CatBoost model at {model_cls_path}")
        LOGGER.info("Loading model instead of retraining...")
        model_cls = cb.CatBoostClassifier()
        model_cls.load_model(str(model_cls_path))
    else:
        LOGGER.info("No cached model found. Training from scratch...")
        
        # Handle class imbalance with class weights
        class_weights = compute_class_weight(
            "balanced",
            classes=np.array([0, 1]),
            y=y_cls_train
        )
        sample_weights = np.where(y_cls_train == 1, class_weights[1], class_weights[0])
        
        LOGGER.info(f"Class weights: {class_weights}")
        LOGGER.info(f"Positive class weight: {class_weights[1]:.2f}")
        
        # Train CatBoost with tuned hyperparameters
        # These are optimized for imbalanced classification with ~3% positive class
        model_cls = cb.CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            
            # Tree structure - deeper trees for complex patterns
            depth=6,  # 6-8 is good for tabular data, avoid overfitting
            l2_leaf_reg=3.0,  # L2 regularization to prevent overfitting
            
            # Learning rate and iterations
            learning_rate=0.03,  # Lower LR with more iterations = better generalization
            iterations=3000,  # More iterations with early stopping
            early_stopping_rounds=150,  # Stop if no improvement for 150 rounds
            
            # Sampling to speed up and reduce overfitting
            subsample=0.8,  # Use 80% of data per iteration
            rsm=0.8,  # Random subspace method: use 80% of features
            
            # Boosting settings
            bootstrap_type="Bernoulli",  # Faster than Bayesian
            
            # Class imbalance handling (already using class_weights)
            auto_class_weights="Balanced",  # Additional balancing
            
            # Performance
            verbose=100,  # Log every 100 iterations
            random_state=42,
            task_type="CPU",
            thread_count=-1  # Use all CPU cores
        )
        
        LOGGER.info("Training CatBoost classifier...")
        model_cls.fit(
            X_train, y_cls_train,
            eval_set=(X_val, y_cls_val),
            sample_weight=sample_weights,
            verbose=False
        )
        
        # Save model
        model_cls.save_model(str(model_cls_path))
        LOGGER.info(f"✅ Saved CatBoost classifier to {model_cls_path}")
    
    # Predict probabilities (soft labels)
    LOGGER.info("Generating soft labels from classifier...")
    p_buyer_train = model_cls.predict_proba(X_train)[:, 1]
    p_buyer_val = model_cls.predict_proba(X_val)[:, 1]
    
    # Evaluate
    metrics_cls = evaluate_classifier(y_cls_val, p_buyer_val, prefix="val_")
    print_metrics(metrics_cls, "CatBoost Classifier - Validation Results")
    
    # =========================================================================
    # 5. Train Teacher Regressor (LightGBM with HistOS)
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 5: Training LightGBM Revenue Regressor with HistOS")
    LOGGER.info("=" * 80)
    
    model_reg_path = MODELS_DIR / "teacher_regressor_lgb_d7.txt"
    
    # Check if model already exists
    if model_reg_path.exists():
        LOGGER.info(f"✅ Found cached LightGBM model at {model_reg_path}")
        LOGGER.info("Loading model instead of retraining...")
        model_reg = lgb.Booster(model_file=str(model_reg_path))
    else:
        LOGGER.info("No cached model found. Training from scratch...")
        
        # Apply HistOS sampling to training data
        LOGGER.info("Applying HistOS sampling...")
        train_df_for_reg = pd.DataFrame(X_train).copy()
        train_df_for_reg["iap_revenue_d7"] = y_reg_train
        
        train_sampled = histos_sample(
            train_df_for_reg,
            revenue_col="iap_revenue_d7",
            bins=[0, 1, 3, 6, 10, np.inf],
            weights=[0.3, 1.0, 2.0, 3.0, 10.0]
        )
        
        X_train_sampled = train_sampled.drop(columns=["iap_revenue_d7"])
        log_rev_train_sampled = np.log1p(train_sampled["iap_revenue_d7"].values)
        
        # Train LightGBM
        train_data = lgb.Dataset(X_train_sampled, label=log_rev_train_sampled)
        val_data = lgb.Dataset(X_val, label=log_rev_val, reference=train_data)
        
        # Tuned LightGBM parameters for revenue regression
        params = {
            "objective": "regression",
            "metric": "rmse",
            
            # Tree structure
            "num_leaves": 63,  # 2^6-1, good balance for tabular data
            "max_depth": 8,  # Prevent overfitting
            "min_child_samples": 20,  # Minimum samples per leaf
            
            # Learning rate and regularization
            "learning_rate": 0.02,  # Lower LR for better generalization
            "lambda_l1": 0.1,  # L1 regularization
            "lambda_l2": 1.0,  # L2 regularization
            
            # Sampling for speed and robustness
            "feature_fraction": 0.8,  # Use 80% of features per tree
            "bagging_fraction": 0.8,  # Use 80% of data per iteration
            "bagging_freq": 5,  # Bagging every 5 iterations
            
            # Performance
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1  # Use all CPU cores
        }
        
        LOGGER.info("Training LightGBM regressor...")
        model_reg = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=200)
            ]
        )
        
        # Save model
        model_reg.save_model(str(model_reg_path))
        LOGGER.info(f"✅ Saved LightGBM regressor to {model_reg_path}")
    
    # Predict log-revenue (soft labels)
    LOGGER.info("Generating soft labels from regressor...")
    log_rev_pred_train = model_reg.predict(X_train, num_iteration=model_reg.best_iteration)
    log_rev_pred_val = model_reg.predict(X_val, num_iteration=model_reg.best_iteration)
    
    # Apply two-stage prediction: non-buyers get zero revenue
    LOGGER.info("Applying two-stage prediction (non-buyers get zero revenue)...")
    rev_pred_train = apply_two_stage_prediction(p_buyer_train, log_rev_pred_train)
    rev_pred_val = apply_two_stage_prediction(p_buyer_val, log_rev_pred_val)
    
    LOGGER.info(f"Train: {(rev_pred_train == 0).sum():,} zeros ({(rev_pred_train == 0).mean():.2%})")
    LOGGER.info(f"Val: {(rev_pred_val == 0).sum():,} zeros ({(rev_pred_val == 0).mean():.2%})")
    
    # Evaluate regressor (on two-stage predictions)
    metrics_reg = evaluate_regressor(y_reg_val, rev_pred_val, prefix="val_")
    print_metrics(metrics_reg, "LightGBM Regressor (Two-Stage) - Validation Results")
    
    # =========================================================================
    # 6. Save Teacher Outputs (Soft Labels)
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 6: Saving Teacher Outputs")
    LOGGER.info("=" * 80)
    
    teacher_outputs_dir = PROCESSED_DIR / "teacher_outputs"
    teacher_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save classifier outputs
    np.savez(
        teacher_outputs_dir / "teacher_classifier_outputs.npz",
        p_buyer_train=p_buyer_train,
        p_buyer_val=p_buyer_val,
        y_train=y_cls_train,
        y_val=y_cls_val
    )
    LOGGER.info(f"✅ Saved classifier outputs")
    
    # Save regressor outputs
    np.savez(
        teacher_outputs_dir / "teacher_regressor_outputs.npz",
        log_rev_train=log_rev_pred_train,
        log_rev_val=log_rev_pred_val,
        y_train=log_rev_train,
        y_val=log_rev_val
    )
    LOGGER.info(f"✅ Saved regressor outputs")
    
    # =========================================================================
    # 7. Evaluate on Test Data (Optional)
    # =========================================================================
    if args.eval_test:
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("STEP 7: Evaluating on Test Data (Memory-Efficient Batch Processing)")
        LOGGER.info("=" * 80)
        LOGGER.info("Loading test data (Oct 8-11)...")

        # Update config for test data
        test_config = deepcopy(config)
        test_config["data"] = dict(config["data"])
        test_config["data"]["test_start"] = "2025-10-08-00-00"
        test_config["data"]["test_end"] = "2025-10-11-23-00"

        test_loader = DataLoader(test_config)
        ddf_test = test_loader.load_test()

        LOGGER.info(f"Test partitions: {ddf_test.npartitions}")

        # Use the same lookup tables from training (don't regenerate)
        lookup_tables_test = lookup_tables

        def build_feature_graph(ddf_source: dd.DataFrame, label: str) -> tuple[dd.DataFrame, str]:
            LOGGER.info(
                "Building test features for %s (lazy compute, %s partitions)...",
                label,
                ddf_source.npartitions,
            )
            X_dask, _ = build_offline_features(
                ddf_source,
                lookup_tables=lookup_tables_test,
                encoders_from_online=encoders,
            )
            LOGGER.info(
                "✅ Feature graph ready for %s (%s partitions)",
                label,
                X_dask.npartitions,
            )
            return X_dask, label

        eval_label = "official test window"
        X_test_dask, current_label = build_feature_graph(ddf_test, eval_label)

        buyer_col_present = "buyer_d7" in ddf_test.columns
        revenue_col_present = "iap_revenue_d7" in ddf_test.columns
        if not (buyer_col_present and revenue_col_present):
            LOGGER.warning(
                "Official test set lacks targets (buyer_d7/iap_revenue_d7). Using fallback window from training data for evaluation."
            )
            fallback_start = args.eval_fallback_start or config["data"]["val_start"]
            fallback_end = args.eval_fallback_end or config["data"]["val_end"]
            ddf_test, current_label = load_labeled_eval_data(test_config, fallback_start, fallback_end)
            LOGGER.info("Loaded fallback labeled slice: %s", current_label)
            X_test_dask, current_label = build_feature_graph(ddf_test, current_label)

        # Process test data in batches to avoid OOM
        MAX_PARTITIONS_PER_BATCH = 4
        n_partitions = X_test_dask.npartitions
        partitions_per_batch = min(MAX_PARTITIONS_PER_BATCH, n_partitions)
        total_batches = (n_partitions + partitions_per_batch - 1) // partitions_per_batch
        LOGGER.info(
            "Processing %s partitions (%s) in batches of %s partitions each (%s batches)...",
            n_partitions,
            current_label,
            partitions_per_batch,
            total_batches,
        )

        # Collect predictions and targets in batches
        all_p_buyer: list[np.ndarray] = []
        all_rev_pred: list[np.ndarray] = []
        all_y_cls: list[np.ndarray] = []
        all_y_reg: list[np.ndarray] = []

        for batch_idx, batch_start in enumerate(range(0, n_partitions, partitions_per_batch), start=1):
            batch_end = min(batch_start + partitions_per_batch, n_partitions)
            LOGGER.info(
                "\nProcessing batch %s/%s (partitions %s-%s)...",
                batch_idx,
                total_batches,
                batch_start,
                batch_end - 1,
            )

            # Get batch of partitions
            batch_partitions = list(range(batch_start, batch_end))
            X_batch_dask = X_test_dask.partitions[batch_partitions]
            ddf_batch = ddf_test.partitions[batch_partitions]

            # Compute this batch
            LOGGER.info("  Computing features for batch %s...", batch_idx)
            X_batch = X_batch_dask.compute()

            # Filter to numeric columns
            numeric_cols = X_batch.select_dtypes(include=[np.number]).columns.tolist()
            X_batch = X_batch[numeric_cols]
            LOGGER.info(f"  Batch {batch_idx}: {X_batch.shape}")

            # Get targets for this batch
            y_cls_batch = ddf_batch["buyer_d7"].compute().values
            y_reg_batch = ddf_batch["iap_revenue_d7"].compute().values

            # Generate predictions for this batch
            LOGGER.info(f"  Generating predictions for batch {batch_idx}...")
            p_buyer_batch = model_cls.predict_proba(X_batch)[:, 1]
            log_rev_pred_batch = model_reg.predict(X_batch, num_iteration=model_reg.best_iteration)

            # Apply two-stage prediction
            rev_pred_batch = apply_two_stage_prediction(p_buyer_batch, log_rev_pred_batch)

            # Collect results
            all_p_buyer.append(p_buyer_batch)
            all_rev_pred.append(rev_pred_batch)
            all_y_cls.append(y_cls_batch)
            all_y_reg.append(y_reg_batch)

            LOGGER.info(f"  ✅ Batch {batch_idx} complete: {len(y_cls_batch):,} samples")

            # Free memory
            del X_batch, y_cls_batch, y_reg_batch, p_buyer_batch, log_rev_pred_batch, rev_pred_batch
            gc.collect()

        # Concatenate all batches
        LOGGER.info("\nConcatenating all batch results (%s batches)...", total_batches)
        p_buyer_test = np.concatenate(all_p_buyer)
        rev_pred_test = np.concatenate(all_rev_pred)
        y_cls_test = np.concatenate(all_y_cls)
        y_reg_test = np.concatenate(all_y_reg)

        LOGGER.info(f"✅ Total evaluation samples ({current_label}): {len(y_cls_test):,}")
        LOGGER.info(f"Test: {(rev_pred_test == 0).sum():,} zeros ({(rev_pred_test == 0).mean():.2%})")

        # Evaluate classifier
        metrics_cls_test = evaluate_classifier(y_cls_test, p_buyer_test, prefix="test_")
        print_metrics(metrics_cls_test, "CatBoost Classifier - Test Results")

        # Evaluate regressor (on two-stage predictions)
        metrics_reg_test = evaluate_regressor(y_reg_test, rev_pred_test, prefix="test_")
        print_metrics(metrics_reg_test, "LightGBM Regressor (Two-Stage) - Test Results")

        # Evaluate two-stage model
        from src.utils.metrics import evaluate_two_stage

        metrics_two_stage_test = evaluate_two_stage(
            y_reg_test, y_cls_test, p_buyer_test, rev_pred_test, prefix="test_"
        )
        print_metrics(metrics_two_stage_test, "Two-Stage Model - Test Results")

        LOGGER.info(f"🎯 TEST MSLE (Competition Metric): {metrics_reg_test['test_msle']:.6f}")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("TRAINING COMPLETE!")
    LOGGER.info("=" * 80)
    LOGGER.info(f"\nModels saved to: {MODELS_DIR}")
    LOGGER.info(f"  - teacher_classifier_catboost.cbm")
    LOGGER.info(f"  - teacher_regressor_lgb_d7.txt")
    LOGGER.info(f"\nSoft labels saved to: {teacher_outputs_dir}")
    LOGGER.info(f"  - teacher_classifier_outputs.npz")
    LOGGER.info(f"  - teacher_regressor_outputs.npz")
    LOGGER.info(f"\nLookup tables saved to: {lookup_dir}")
    LOGGER.info(f"\nNext step: Run 'uv run python scripts/train_students.py' to train student models")


if __name__ == "__main__":
    main()

