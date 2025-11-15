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
from pathlib import Path

import catboost as cb
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
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "teachers"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train teacher models")
    parser.add_argument(
        "--mode",
        choices=["subset", "full"],
        default="subset",
        help="Training mode: 'subset' (Oct 1 only, fast) or 'full' (Oct 1-5, slow)"
    )
    # Backward compatibility
    parser.add_argument("--subset", action="store_const", const="subset", dest="mode")
    parser.add_argument("--full", action="store_const", const="full", dest="mode")
    
    return parser.parse_args()


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
    
    # Time-based split configuration
    if args.mode == "subset":
        # Fast iteration: Oct 1 only for training
        LOGGER.info("Loading SUBSET (Oct 1 only) for fast iteration...")
        train_start = "2025-10-01-00-00"
        train_end = "2025-10-01-23-00"
    else:
        # Full training: Oct 1-5
        LOGGER.info("Loading FULL dataset (Oct 1-5)...")
        train_start = "2025-10-01-00-00"
        train_end = "2025-10-05-23-00"
    
    # Validation: Oct 6
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
    import pickle
    encoder_path = PROCESSED_DIR / "encoders" / "online_encoders.pkl"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoders, f)
    LOGGER.info(f"✅ Saved encoders to {encoder_path}")
    
    # Compute features (bring into memory)
    LOGGER.info("Computing features (this may take a few minutes)...")
    X_train = X_train_dask.compute()
    X_val = X_val_dask.compute()
    
    # Get targets
    LOGGER.info("Extracting targets...")
    y_cls_train = ddf_train["buyer_d7"].compute().values
    y_cls_val = ddf_val["buyer_d7"].compute().values
    
    y_reg_train = ddf_train["iap_revenue_d7"].compute().values
    y_reg_val = ddf_val["iap_revenue_d7"].compute().values
    
    # Convert to log scale for regressor
    log_rev_train = np.log1p(y_reg_train)
    log_rev_val = np.log1p(y_reg_val)
    
    LOGGER.info(f"Train shape: {X_train.shape}")
    LOGGER.info(f"Val shape: {X_val.shape}")
    LOGGER.info(f"Buyer rate (train): {y_cls_train.mean():.4f}")
    LOGGER.info(f"Buyer rate (val): {y_cls_val.mean():.4f}")
    
    # =========================================================================
    # 4. Train Teacher Classifier (CatBoost)
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 4: Training CatBoost Buyer Classifier")
    LOGGER.info("=" * 80)
    
    # Handle class imbalance with class weights
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1]),
        y=y_cls_train
    )
    sample_weights = np.where(y_cls_train == 1, class_weights[1], class_weights[0])
    
    LOGGER.info(f"Class weights: {class_weights}")
    LOGGER.info(f"Positive class weight: {class_weights[1]:.2f}")
    
    # Train CatBoost
    model_cls = cb.CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=8,
        learning_rate=0.05,
        iterations=2000,
        early_stopping_rounds=100,
        verbose=200,
        random_state=42,
        task_type="CPU"
    )
    
    LOGGER.info("Training CatBoost classifier...")
    model_cls.fit(
        X_train, y_cls_train,
        eval_set=(X_val, y_cls_val),
        sample_weight=sample_weights,
        verbose=False
    )
    
    # Save model
    model_cls_path = MODELS_DIR / "teacher_classifier_catboost.cbm"
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
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 127,
        "max_depth": 10,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1
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
    model_reg_path = MODELS_DIR / "teacher_regressor_lgb_d7.txt"
    model_reg.save_model(str(model_reg_path))
    LOGGER.info(f"✅ Saved LightGBM regressor to {model_reg_path}")
    
    # Predict log-revenue (soft labels)
    LOGGER.info("Generating soft labels from regressor...")
    log_rev_pred_train = model_reg.predict(X_train, num_iteration=model_reg.best_iteration)
    log_rev_pred_val = model_reg.predict(X_val, num_iteration=model_reg.best_iteration)
    
    # Convert back to original scale for evaluation
    rev_pred_val = np.expm1(log_rev_pred_val)
    
    # Evaluate
    metrics_reg = evaluate_regressor(y_reg_val, rev_pred_val, prefix="val_")
    print_metrics(metrics_reg, "LightGBM Regressor - Validation Results")
    
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
    
    # Close Dask client
    loader.close()


if __name__ == "__main__":
    main()

