#!/usr/bin/env python3
"""Generate predictions using two-stage LightGBM model.

Usage:
    uv run python scripts/predict_lgb_two_stage.py

This script:
1. Loads test data from the competition
2. Loads trained models (classifier + regressor)
3. Generates predictions using two-stage approach
4. Creates submission.csv with exactly the required number of rows
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pickle

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.features.offline import build_offline_features
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "teachers"  # Use teacher models
SUBMISSIONS_DIR = DATA_DIR / "submissions"

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


def create_test_config() -> dict:
    """Create configuration for test data loading.
    
    Returns:
        Configuration dictionary with test date range
    """
    return {
        "data": {
            "train_path": str(DATA_DIR / "raw" / "train" / "train"),
            "test_path": str(DATA_DIR / "raw" / "test" / "test"),
            "test_start": "2025-10-08-00-00",
            "test_end": "2025-10-11-23-00",
        }
    }


def load_models() -> tuple[cb.CatBoostClassifier, lgb.Booster, dict]:
    """Load trained models and lookup tables.
    
    Returns:
        Tuple of (classifier, regressor, lookup_tables)
    """
    classifier_path = MODELS_DIR / "teacher_classifier_catboost.cbm"
    regressor_path = MODELS_DIR / "teacher_regressor_lgb_d7.txt"
    lookup_path = DATA_DIR / "processed" / "lookup_tables"  # Directory, not pickle file
    
    # Check all files exist
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier not found: {classifier_path}")
    if not regressor_path.exists():
        raise FileNotFoundError(f"Regressor not found: {regressor_path}")
    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup tables not found: {lookup_path}")
    
    LOGGER.info("Loading models...")
    classifier = cb.CatBoostClassifier()
    classifier.load_model(str(classifier_path))
    LOGGER.info(f"✅ Classifier loaded (CatBoost)")
    
    regressor = lgb.Booster(model_file=str(regressor_path))
    LOGGER.info(f"✅ Regressor loaded: {regressor.num_trees()} trees")
    
    # Load lookup tables from JSON files
    import json
    lookup_tables = {}
    for lookup_file in lookup_path.glob("*.json"):
        with open(lookup_file) as f:
            lookup_tables[lookup_file.stem] = json.load(f)
    LOGGER.info(f"✅ Lookup tables loaded: {list(lookup_tables.keys())}")
    LOGGER.info("✅ Lookup tables loaded")
    
    return classifier, regressor, lookup_tables


def predict_batch(
    classifier: lgb.Booster,
    regressor: lgb.Booster,
    X_batch: pd.DataFrame,
) -> np.ndarray:
    """Generate predictions for a batch using two-stage approach.
    
    Args:
        classifier: Trained buyer classifier
        regressor: Trained revenue regressor
        X_batch: Feature batch
        
    Returns:
        Revenue predictions for the batch
        
    Postconditions:
        - Non-buyers (classifier < 0.5) always get exactly 0.0 revenue
        - All predictions are non-negative
    """
    # Stage 1: Predict buyers
    buyer_proba = classifier.predict(X_batch)
    buyer_pred = (buyer_proba > 0.5).astype(bool)
    
    # Stage 2: Initialize all predictions to 0 (non-buyers get zero revenue)
    predictions = np.zeros(len(X_batch), dtype=np.float64)
    
    # Only predict revenue for predicted buyers
    n_buyers = buyer_pred.sum()
    if n_buyers > 0:
        X_buyers = X_batch[buyer_pred]
        log_revenue = regressor.predict(X_buyers)
        revenue = np.expm1(log_revenue)
        revenue = np.maximum(revenue, 0.0)  # Clip negative predictions
        predictions[buyer_pred] = revenue
    
    # Postcondition: verify non-buyers have zero revenue
    assert np.all(predictions[~buyer_pred] == 0.0), "Non-buyers must have zero revenue"
    assert np.all(predictions >= 0.0), "All predictions must be non-negative"
    
    return predictions


def main() -> None:
    """Main prediction pipeline."""
    LOGGER.info("=" * 80)
    LOGGER.info("TWO-STAGE LIGHTGBM PREDICTION")
    LOGGER.info("=" * 80)
    
    # Load models
    classifier, regressor, lookup_tables = load_models()
    
    # Load test data
    LOGGER.info("\nLoading test data...")
    config = create_test_config()
    loader = DataLoader(config)
    ddf_test = loader.load_test()
    
    LOGGER.info(f"Test data: {ddf_test.npartitions} partitions")
    
    # Check for row_id before computing features
    if "row_id" not in ddf_test.columns:
        raise ValueError("Test data missing 'row_id' column!")
    
    # Extract row_ids first (before feature engineering modifies the dataframe)
    LOGGER.info("Extracting row IDs...")
    row_ids = ddf_test["row_id"].compute()
    n_test_rows = len(row_ids)
    LOGGER.info(f"Test samples: {n_test_rows:,}")
    
    # Build features
    LOGGER.info("\nBuilding test features...")
    X_test_dask, _ = build_offline_features(
        ddf_test,
        lookup_tables=lookup_tables,
        encoders_from_online=None,  # Will create new encoders for test
    )
    
    LOGGER.info(f"Computing test features ({X_test_dask.npartitions} partitions)...")
    X_test = X_test_dask.compute()
    LOGGER.info(f"✅ Test features: {X_test.shape}")
    
    # Verify row count matches
    assert len(X_test) == n_test_rows, f"Feature count mismatch: {len(X_test)} != {n_test_rows}"
    
    # Generate predictions
    LOGGER.info("\nGenerating predictions...")
    predictions = predict_batch(classifier, regressor, X_test)
    
    LOGGER.info(f"\nPrediction statistics:")
    LOGGER.info(f"  Count: {len(predictions):,}")
    LOGGER.info(f"  Mean: ${predictions.mean():.2f}")
    LOGGER.info(f"  Median: ${np.median(predictions):.2f}")
    LOGGER.info(f"  Min: ${predictions.min():.2f}")
    LOGGER.info(f"  Max: ${predictions.max():.2f}")
    LOGGER.info(f"  Zeros: {(predictions == 0).sum():,} ({(predictions == 0).mean():.2%})")
    LOGGER.info(f"  Non-zeros: {(predictions > 0).sum():,} ({(predictions > 0).mean():.2%})")
    
    # Create submission
    LOGGER.info("\nCreating submission file...")
    submission = pd.DataFrame({
        "row_id": row_ids,
        "iap_revenue_d7": predictions,
    })
    
    # Validate submission format
    LOGGER.info("Validating submission...")
    assert len(submission) == n_test_rows, f"Submission length mismatch: {len(submission)} != {n_test_rows}"
    assert "row_id" in submission.columns, "Missing row_id column"
    assert "iap_revenue_d7" in submission.columns, "Missing iap_revenue_d7 column"
    assert submission["iap_revenue_d7"].notna().all(), "NaN values found in predictions"
    assert (submission["iap_revenue_d7"] >= 0).all(), "Negative predictions found"
    assert submission["row_id"].notna().all(), "NaN values found in row_id"
    
    # Check for expected row count (13188409 per user requirement)
    EXPECTED_ROWS = 13188409
    if len(submission) != EXPECTED_ROWS:
        LOGGER.warning(f"⚠️  Expected {EXPECTED_ROWS:,} rows, got {len(submission):,}")
    else:
        LOGGER.info(f"✅ Row count matches expected: {EXPECTED_ROWS:,}")
    
    LOGGER.info("✅ Submission validated")
    
    # Save submission
    submission_path = SUBMISSIONS_DIR / "submission_lgb_two_stage.csv"
    submission.to_csv(submission_path, index=False)
    
    file_size_mb = submission_path.stat().st_size / (1024 * 1024)
    LOGGER.info(f"\n✅ Submission saved: {submission_path}")
    LOGGER.info(f"   Rows: {len(submission):,}")
    LOGGER.info(f"   Size: {file_size_mb:.1f} MB")
    
    # Show sample
    LOGGER.info("\nFirst 10 rows:")
    print(submission.head(10).to_string(index=False))
    
    LOGGER.info("\nLast 10 rows:")
    print(submission.tail(10).to_string(index=False))
    
    # Summary
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("PREDICTION COMPLETE!")
    LOGGER.info("=" * 80)
    LOGGER.info(f"✅ Submission file: {submission_path}")
    LOGGER.info(f"📊 Rows: {len(submission):,}")
    LOGGER.info(f"📊 Predicted buyers: {(predictions > 0).sum():,} ({(predictions > 0).mean():.2%})")
    LOGGER.info(f"📊 Mean revenue: ${predictions.mean():.2f}")


if __name__ == "__main__":
    main()
