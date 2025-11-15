#!/usr/bin/env python3
"""Simplified fast teacher training - pandas-based for speed.

This version loads data and immediately converts to pandas for fast processing.
Trades some scalability for massive speed improvement on subset data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.metrics import evaluate_classifier, evaluate_regressor, print_metrics

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "teachers"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train teacher models (FAST version)")
    parser.add_argument("--subset", action="store_true", help="Use Oct 1 only (fast)")
    return parser.parse_args()


def load_parquet_simple(path: Path, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load parquet directly with pandas - much faster for small datasets."""
    LOGGER.info(f"Loading from {path}...")
    
    # Find matching partition directories
    partitions = []
    for datetime_dir in sorted(path.glob("datetime=*")):
        datetime_val = datetime_dir.name.replace("datetime=", "")
        
        # Check if datetime is in range
        include = True
        if start_date and datetime_val < start_date:
            include = False
        if end_date and datetime_val >= end_date:
            include = False
        
        if include:
            partitions.append(datetime_dir)
    
    LOGGER.info(f"Found {len(partitions)} matching partitions")
    
    # Read all partitions
    dfs = []
    for i, partition_dir in enumerate(partitions):
        if i % 5 == 0:
            LOGGER.info(f"  Loading partition {i+1}/{len(partitions)}...")
        
        for parquet_file in partition_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet_file)
            df['datetime'] = partition_dir.name.replace("datetime=", "")
            dfs.append(df)
    
    result = pd.concat(dfs, ignore_index=True)
    LOGGER.info(f"✅ Loaded {len(result):,} rows, {len(result.columns)} columns")
    
    return result


def build_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build simple numeric features only - skip complex nested columns."""
    LOGGER.info("Building simple features...")
    
    # Keep only numeric and simple categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Simple string columns that are easy to encode
    simple_cat_cols = [
        'advertiser_category', 'advertiser_subcategory', 'country', 
        'dev_os', 'dev_make', 'hour'
    ]
    
    # Select features
    feature_cols = numeric_cols + [c for c in simple_cat_cols if c in df.columns]
    X = df[feature_cols].copy()
    
    # Encode categoricals
    LOGGER.info("Encoding categorical features...")
    encoders = {}
    for col in simple_cat_cols:
        if col in X.columns:
            le = LabelEncoder()
            mask = X[col].notna()
            X.loc[mask, col] = le.fit_transform(X.loc[mask, col].astype(str))
            X[col] = X[col].fillna(-1).astype(int)
            encoders[col] = le
    
    # Fill missing
    X = X.fillna(0)
    
    LOGGER.info(f"✅ Features built: {X.shape}")
    
    return X, encoders


def main():
    args = parse_args()
    
    LOGGER.info("=" * 80)
    LOGGER.info("FAST TEACHER TRAINING PIPELINE")
    LOGGER.info("=" * 80)
    
    # Load data
    LOGGER.info("\nSTEP 1: Loading Data")
    LOGGER.info("=" * 80)
    
    train_path = DATA_DIR / "raw" / "train" / "train"
    
    if args.subset:
        LOGGER.info("Loading SUBSET (Oct 1 only)")
        df_train = load_parquet_simple(train_path, "2025-10-01", "2025-10-02")
        df_val = load_parquet_simple(train_path, "2025-10-06", "2025-10-07")
    else:
        LOGGER.info("Loading FULL dataset")
        df_train = load_parquet_simple(train_path, "2025-10-01", "2025-10-06")
        df_val = load_parquet_simple(train_path, "2025-10-06", "2025-10-07")
    
    # Build features
    LOGGER.info("\nSTEP 2: Building Features")
    LOGGER.info("=" * 80)
    
    X_train, encoders = build_simple_features(df_train)
    X_val, _ = build_simple_features(df_val)
    
    # Align columns
    train_cols = set(X_train.columns)
    val_cols = set(X_val.columns)
    common_cols = sorted(train_cols & val_cols)
    
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    
    # Targets
    y_cls_train = df_train['buyer_d7'].values
    y_cls_val = df_val['buyer_d7'].values
    
    y_reg_train = df_train['iap_revenue_d7'].values
    y_reg_val = df_val['iap_revenue_d7'].values
    
    log_rev_train = np.log1p(y_reg_train)
    log_rev_val = np.log1p(y_reg_val)
    
    LOGGER.info(f"Train: {X_train.shape}, Buyer rate: {y_cls_train.mean():.4f}")
    LOGGER.info(f"Val: {X_val.shape}, Buyer rate: {y_cls_val.mean():.4f}")
    
    # Train classifier
    LOGGER.info("\nSTEP 3: Training CatBoost Classifier")
    LOGGER.info("=" * 80)
    
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_cls_train)
    sample_weights = np.where(y_cls_train == 1, class_weights[1], class_weights[0])
    
    model_cls = cb.CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        depth=6,
        learning_rate=0.1,
        iterations=500,
        early_stopping_rounds=50,
        verbose=100,
        random_state=42
    )
    
    model_cls.fit(
        X_train, y_cls_train,
        eval_set=(X_val, y_cls_val),
        sample_weight=sample_weights
    )
    
    model_cls.save_model(str(MODELS_DIR / "teacher_classifier_catboost.cbm"))
    
    p_buyer_train = model_cls.predict_proba(X_train)[:, 1]
    p_buyer_val = model_cls.predict_proba(X_val)[:, 1]
    
    metrics_cls = evaluate_classifier(y_cls_val, p_buyer_val, prefix="val_")
    print_metrics(metrics_cls, "Classifier Validation")
    
    # Train regressor
    LOGGER.info("\nSTEP 4: Training LightGBM Regressor")
    LOGGER.info("=" * 80)
    
    model_reg = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=63,
        max_depth=8,
        learning_rate=0.05,
        n_estimators=500,
        verbose=-1,
        random_state=42
    )
    
    model_reg.fit(
        X_train, log_rev_train,
        eval_set=[(X_val, log_rev_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    model_reg.booster_.save_model(str(MODELS_DIR / "teacher_regressor_lgb_d7.txt"))
    
    log_rev_pred_train = model_reg.predict(X_train)
    log_rev_pred_val = model_reg.predict(X_val)
    
    rev_pred_val = np.expm1(log_rev_pred_val)
    metrics_reg = evaluate_regressor(y_reg_val, rev_pred_val, prefix="val_")
    print_metrics(metrics_reg, "Regressor Validation")
    
    # Save outputs
    LOGGER.info("\nSTEP 5: Saving Outputs")
    LOGGER.info("=" * 80)
    
    outputs_dir = PROCESSED_DIR / "teacher_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        outputs_dir / "teacher_classifier_outputs.npz",
        p_buyer_train=p_buyer_train,
        p_buyer_val=p_buyer_val,
        y_train=y_cls_train,
        y_val=y_cls_val
    )
    
    np.savez(
        outputs_dir / "teacher_regressor_outputs.npz",
        log_rev_train=log_rev_pred_train,
        log_rev_val=log_rev_pred_val,
        y_train=log_rev_train,
        y_val=log_rev_val
    )
    
    # Save encoders
    import pickle
    encoder_path = PROCESSED_DIR / "encoders" / "simple_encoders.pkl"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    with open(encoder_path, 'wb') as f:
        pickle.dump({'encoders': encoders, 'feature_names': common_cols}, f)
    
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("✅ TRAINING COMPLETE!")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Models saved to: {MODELS_DIR}")
    LOGGER.info(f"Next: Train students with similar simple approach")


if __name__ == "__main__":
    main()

