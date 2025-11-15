#!/usr/bin/env python3
"""Generate submission file for competition.

Usage:
    uv run python scripts/make_submission.py

This script:
1. Loads test data
2. Loads student models and artifacts
3. Generates predictions using fast inference
4. Creates submission.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import DataLoader
from src.inference.predictor import RevenuePredictor
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = DATA_DIR / "submissions"

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Main submission generation pipeline."""
    LOGGER.info("=" * 80)
    LOGGER.info("SUBMISSION GENERATION")
    LOGGER.info("=" * 80)
    
    # =========================================================================
    # 1. Load Test Data
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 1: Loading Test Data")
    LOGGER.info("=" * 80)
    
    config = {
        "data": {
            "train_path": str(DATA_DIR / "raw" / "train" / "train"),
            "test_path": str(DATA_DIR / "raw" / "test" / "test"),
            "train_start": "2025-10-01-00-00",
            "train_end": "2025-10-05-23-00",
            "val_start": "2025-10-06-00-00",
            "val_end": "2025-10-06-23-00",
            "test_start": "2025-10-07-00-00",
            "test_end": "2025-10-07-23-00"
        },
        "dask": {
            "client": {
                "enabled": True,
                "n_workers": 4,
                "threads_per_worker": 4
            }
        }
    }
    
    loader = DataLoader(config)
    ddf_test = loader.load_test()
    
    LOGGER.info(f"Test data: {ddf_test.npartitions} partitions")
    
    # Compute (test set is usually small enough)
    LOGGER.info("Computing test data...")
    df_test = ddf_test.compute()
    
    LOGGER.info(f"Test data shape: {df_test.shape}")
    
    # Check for row_id column
    if "row_id" not in df_test.columns:
        LOGGER.error("❌ Test data missing 'row_id' column!")
        raise ValueError("Test data must have 'row_id' column for submission")
    
    LOGGER.info(f"Test samples: {len(df_test):,}")
    
    # =========================================================================
    # 2. Initialize Predictor
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 2: Initializing Predictor")
    LOGGER.info("=" * 80)
    
    predictor = RevenuePredictor(project_root=PROJECT_ROOT)
    
    # =========================================================================
    # 3. Generate Predictions
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 3: Generating Predictions")
    LOGGER.info("=" * 80)
    
    predictions = predictor.predict(df_test)
    
    LOGGER.info(f"\nPrediction statistics:")
    LOGGER.info(f"  Mean: ${predictions.mean():.2f}")
    LOGGER.info(f"  Median: ${predictions.median():.2f}")
    LOGGER.info(f"  Min: ${predictions.min():.2f}")
    LOGGER.info(f"  Max: ${predictions.max():.2f}")
    LOGGER.info(f"  Zeros: {(predictions == 0).sum():,} ({(predictions == 0).mean():.2%})")
    
    # =========================================================================
    # 4. Create Submission
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 4: Creating Submission File")
    LOGGER.info("=" * 80)
    
    submission = pd.DataFrame({
        "row_id": df_test["row_id"],
        "iap_revenue_d7": predictions.astype(float)
    })
    
    # Validate submission format
    LOGGER.info("Validating submission format...")
    assert len(submission) == len(df_test), "Submission length mismatch"
    assert "row_id" in submission.columns, "Missing row_id column"
    assert "iap_revenue_d7" in submission.columns, "Missing iap_revenue_d7 column"
    assert submission["iap_revenue_d7"].notna().all(), "NaN values in predictions"
    assert (submission["iap_revenue_d7"] >= 0).all(), "Negative predictions found"
    
    LOGGER.info("✅ Submission format validated")
    
    # Save submission
    submission_path = SUBMISSIONS_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    
    LOGGER.info(f"✅ Submission saved to: {submission_path}")
    LOGGER.info(f"   Rows: {len(submission):,}")
    LOGGER.info(f"   Size: {submission_path.stat().st_size / 1024:.1f} KB")
    
    # Display first few rows
    LOGGER.info("\nFirst 10 rows of submission:")
    print(submission.head(10).to_string(index=False))
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("SUBMISSION GENERATION COMPLETE!")
    LOGGER.info("=" * 80)
    LOGGER.info(f"\nSubmission file: {submission_path}")
    LOGGER.info(f"Predictions: {len(submission):,} rows")
    LOGGER.info(f"\nYou can now submit this file to the competition!")
    
    # Close Dask client
    loader.close()


if __name__ == "__main__":
    main()

