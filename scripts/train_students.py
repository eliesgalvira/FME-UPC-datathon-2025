#!/usr/bin/env python3
"""Train student models via distillation from teachers.

Usage:
    uv run python scripts/train_students.py

This script:
1. Loads training data with train/val split
2. Loads teacher soft labels
3. Builds online features (fast features for students)
4. Trains student classifier (small LightGBM)
5. Trains student regressor (small LightGBM)
6. Saves models for fast inference
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.data.loader import DataLoader
from src.features.lookup_tables import load_lookup_tables
from src.features.online import build_online_features, load_encoders, save_encoders
from src.utils.logger import get_logger
from src.utils.metrics import evaluate_classifier, evaluate_two_stage, print_metrics

LOGGER = get_logger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "students"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Main student training pipeline."""
    LOGGER.info("=" * 80)
    LOGGER.info("STUDENT TRAINING PIPELINE (Distillation)")
    LOGGER.info("=" * 80)
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 1: Loading Data")
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
    ddf_train, ddf_val = loader.load_train(validation_split=True)
    
    # =========================================================================
    # 2. Load Lookup Tables and Encoders
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 2: Loading Lookup Tables and Encoders")
    LOGGER.info("=" * 80)
    
    lookup_tables = load_lookup_tables(PROCESSED_DIR / "lookup_tables")
    encoders = load_encoders(PROCESSED_DIR / "encoders" / "online_encoders.pkl")
    
    # =========================================================================
    # 3. Build Online Features
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 3: Building Online Features (Fast Features)")
    LOGGER.info("=" * 80)
    
    LOGGER.info("Building features for training data...")
    X_stu_train_dask, _ = build_online_features(
        ddf_train,
        lookup_tables=lookup_tables,
        encoders=encoders,
        fit_encoders=False
    )
    
    LOGGER.info("Building features for validation data...")
    X_stu_val_dask, _ = build_online_features(
        ddf_val,
        lookup_tables=lookup_tables,
        encoders=encoders,
        fit_encoders=False
    )
    
    # Compute
    LOGGER.info("Computing features...")
    X_stu_train = X_stu_train_dask.compute()
    X_stu_val = X_stu_val_dask.compute()
    
    # Get ground truth
    y_cls_train = ddf_train["buyer_d7"].compute().values
    y_cls_val = ddf_val["buyer_d7"].compute().values
    
    y_reg_train = np.log1p(ddf_train["iap_revenue_d7"].compute().values)
    y_reg_val = np.log1p(ddf_val["iap_revenue_d7"].compute().values)
    
    LOGGER.info(f"Train shape: {X_stu_train.shape}")
    LOGGER.info(f"Val shape: {X_stu_val.shape}")
    
    # =========================================================================
    # 4. Load Teacher Soft Labels
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 4: Loading Teacher Soft Labels")
    LOGGER.info("=" * 80)
    
    teacher_outputs_dir = PROCESSED_DIR / "teacher_outputs"
    
    # Classifier outputs
    teacher_cls = np.load(teacher_outputs_dir / "teacher_classifier_outputs.npz")
    p_teacher_train = teacher_cls["p_buyer_train"]
    p_teacher_val = teacher_cls["p_buyer_val"]
    
    # Regressor outputs
    teacher_reg = np.load(teacher_outputs_dir / "teacher_regressor_outputs.npz")
    log_rev_teacher_train = teacher_reg["log_rev_train"]
    log_rev_teacher_val = teacher_reg["log_rev_val"]
    
    LOGGER.info(f"Loaded teacher outputs for {len(p_teacher_train):,} train samples")
    LOGGER.info(f"Loaded teacher outputs for {len(p_teacher_val):,} val samples")
    
    # =========================================================================
    # 5. Train Student Classifier
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 5: Training Student Classifier")
    LOGGER.info("=" * 80)
    
    # Blend teacher predictions with ground truth
    alpha = 0.8  # 80% teacher, 20% ground truth
    y_soft_cls_train = alpha * p_teacher_train + (1 - alpha) * y_cls_train
    y_soft_cls_val = alpha * p_teacher_val + (1 - alpha) * y_cls_val
    
    LOGGER.info(f"Soft labels: {alpha:.0%} teacher + {1-alpha:.0%} ground truth")
    
    # Train tiny LightGBM (regression to soft probability)
    train_cls_data = lgb.Dataset(X_stu_train, label=y_soft_cls_train)
    val_cls_data = lgb.Dataset(X_stu_val, label=y_soft_cls_val, reference=train_cls_data)
    
    params_cls = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "max_depth": 5,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1
    }
    
    LOGGER.info("Training student classifier...")
    student_cls = lgb.train(
        params_cls,
        train_cls_data,
        num_boost_round=150,
        valid_sets=[train_cls_data, val_cls_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Save
    model_cls_path = MODELS_DIR / "student_classifier_lgb.txt"
    student_cls.save_model(str(model_cls_path))
    LOGGER.info(f"✅ Saved student classifier to {model_cls_path}")
    
    # Evaluate
    p_student_val = np.clip(
        student_cls.predict(X_stu_val, num_iteration=student_cls.best_iteration),
        0.0, 1.0
    )
    
    metrics_cls = evaluate_classifier(y_cls_val, p_student_val, prefix="student_")
    print_metrics(metrics_cls, "Student Classifier - Validation Results")
    
    # Compare to teacher
    LOGGER.info("\nStudent vs Teacher Classifier:")
    from src.utils.metrics import auc_roc, auc_pr
    LOGGER.info(f"  Teacher AUC: {auc_roc(y_cls_val, p_teacher_val):.4f}")
    LOGGER.info(f"  Student AUC: {auc_roc(y_cls_val, p_student_val):.4f}")
    LOGGER.info(f"  Teacher AUC-PR: {auc_pr(y_cls_val, p_teacher_val):.4f}")
    LOGGER.info(f"  Student AUC-PR: {auc_pr(y_cls_val, p_student_val):.4f}")
    
    # =========================================================================
    # 6. Train Student Regressor
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STEP 6: Training Student Regressor")
    LOGGER.info("=" * 80)
    
    # Blend teacher predictions with ground truth
    y_soft_reg_train = alpha * log_rev_teacher_train + (1 - alpha) * y_reg_train
    y_soft_reg_val = alpha * log_rev_teacher_val + (1 - alpha) * y_reg_val
    
    # Train tiny LightGBM
    train_reg_data = lgb.Dataset(X_stu_train, label=y_soft_reg_train)
    val_reg_data = lgb.Dataset(X_stu_val, label=y_soft_reg_val, reference=train_reg_data)
    
    params_reg = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "max_depth": 5,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1
    }
    
    LOGGER.info("Training student regressor...")
    student_reg = lgb.train(
        params_reg,
        train_reg_data,
        num_boost_round=150,
        valid_sets=[train_reg_data, val_reg_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Save
    model_reg_path = MODELS_DIR / "student_regressor_lgb.txt"
    student_reg.save_model(str(model_reg_path))
    LOGGER.info(f"✅ Saved student regressor to {model_reg_path}")
    
    # Evaluate
    log_rev_student_val = student_reg.predict(X_stu_val, num_iteration=student_reg.best_iteration)
    rev_student_val = np.expm1(log_rev_student_val)
    
    # Two-stage prediction
    final_pred_val = p_student_val * rev_student_val
    rev_true_val = np.expm1(y_reg_val)
    
    from src.utils.metrics import msle, rmse
    LOGGER.info("\nStudent Regressor Performance:")
    LOGGER.info(f"  RMSE (log-scale): {rmse(rev_true_val, rev_student_val, log_scale=True):.4f}")
    LOGGER.info(f"  MSLE (2-stage): {msle(rev_true_val, final_pred_val):.4f}")
    
    # Compare to teacher
    rev_teacher_val = np.expm1(log_rev_teacher_val)
    final_teacher_val = p_teacher_val * rev_teacher_val
    
    LOGGER.info("\nStudent vs Teacher (2-Stage):")
    LOGGER.info(f"  Teacher MSLE: {msle(rev_true_val, final_teacher_val):.4f}")
    LOGGER.info(f"  Student MSLE: {msle(rev_true_val, final_pred_val):.4f}")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("STUDENT TRAINING COMPLETE!")
    LOGGER.info("=" * 80)
    LOGGER.info(f"\nModels saved to: {MODELS_DIR}")
    LOGGER.info(f"  - student_classifier_lgb.txt")
    LOGGER.info(f"  - student_regressor_lgb.txt")
    LOGGER.info(f"\nModel sizes:")
    LOGGER.info(f"  Classifier: {model_cls_path.stat().st_size / 1024:.1f} KB")
    LOGGER.info(f"  Regressor: {model_reg_path.stat().st_size / 1024:.1f} KB")
    LOGGER.info(f"\nNext step: Run 'uv run python scripts/make_submission.py' to generate predictions")
    
    # Close Dask client
    loader.close()


if __name__ == "__main__":
    main()

