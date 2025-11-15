"""Evaluation metrics for revenue prediction.

Metrics implemented:
- MSLE (Mean Squared Log Error): Primary competition metric
- RMSE: Root Mean Squared Error (log-scale and original-scale)
- MAE: Mean Absolute Error
- AUC: Area Under ROC Curve (for buyer classification)
- AUC-PR: Area Under Precision-Recall Curve (for imbalanced classification)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def msle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Log Error.
    
    Primary metric for revenue prediction competition.
    Penalizes relative errors more than absolute errors.
    
    Args:
        y_true: Ground truth revenue (original scale)
        y_pred: Predicted revenue (original scale)
        
    Returns:
        MSLE score (lower is better)
        
    Note:
        Predictions are clipped to >= 0 to avoid log errors.
    """
    y_pred = np.maximum(0, y_pred)  # Clip negative predictions
    return float(mean_squared_log_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray, log_scale: bool = False) -> float:
    """Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        log_scale: If True, compute RMSE on log1p scale
        
    Returns:
        RMSE score (lower is better)
    """
    if log_scale:
        y_true = np.log1p(y_true)
        y_pred = np.log1p(np.maximum(0, y_pred))
    
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray, log_scale: bool = False) -> float:
    """Mean Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        log_scale: If True, compute MAE on log1p scale
        
    Returns:
        MAE score (lower is better)
    """
    if log_scale:
        y_true = np.log1p(y_true)
        y_pred = np.log1p(np.maximum(0, y_pred))
    
    return float(mean_absolute_error(y_true, y_pred))


def auc_roc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Area Under ROC Curve.
    
    Args:
        y_true: Ground truth binary labels (0/1)
        y_pred_proba: Predicted probabilities
        
    Returns:
        AUC score (higher is better, 0.5 = random, 1.0 = perfect)
    """
    return float(roc_auc_score(y_true, y_pred_proba))


def auc_pr(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Area Under Precision-Recall Curve.
    
    Better metric than AUC-ROC for imbalanced datasets.
    
    Args:
        y_true: Ground truth binary labels (0/1)
        y_pred_proba: Predicted probabilities
        
    Returns:
        AUC-PR score (higher is better)
    """
    return float(average_precision_score(y_true, y_pred_proba))


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    prefix: str = ""
) -> Dict[str, float]:
    """Comprehensive evaluation for binary classifier.
    
    Args:
        y_true: Ground truth binary labels (0/1)
        y_pred_proba: Predicted probabilities
        threshold: Decision threshold for binary predictions
        prefix: Prefix for metric names (e.g., "train_", "val_")
        
    Returns:
        Dict of metrics
        
    Example:
        >>> from src.utils.metrics import evaluate_classifier
        >>> metrics = evaluate_classifier(y_val, p_pred_val, prefix="val_")
        >>> print(f"Val AUC: {metrics['val_auc_roc']:.4f}")
    """
    metrics = {}
    
    # AUC scores
    metrics[f"{prefix}auc_roc"] = auc_roc(y_true, y_pred_proba)
    metrics[f"{prefix}auc_pr"] = auc_pr(y_true, y_pred_proba)
    
    # Binary predictions
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    # Precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics[f"{prefix}precision"] = float(precision)
    metrics[f"{prefix}recall"] = float(recall)
    metrics[f"{prefix}f1"] = float(f1)
    metrics[f"{prefix}accuracy"] = float((tp + tn) / len(y_true))
    
    # Class distribution
    metrics[f"{prefix}pos_rate_true"] = float(np.mean(y_true))
    metrics[f"{prefix}pos_rate_pred"] = float(np.mean(y_pred_binary))
    
    return metrics


def evaluate_regressor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """Comprehensive evaluation for revenue regressor.
    
    Args:
        y_true: Ground truth revenue (original scale)
        y_pred: Predicted revenue (original scale)
        prefix: Prefix for metric names (e.g., "train_", "val_")
        
    Returns:
        Dict of metrics
        
    Example:
        >>> from src.utils.metrics import evaluate_regressor
        >>> metrics = evaluate_regressor(rev_val, rev_pred_val, prefix="val_")
        >>> print(f"Val MSLE: {metrics['val_msle']:.4f}")
    """
    metrics = {}
    
    # Clip predictions
    y_pred = np.maximum(0, y_pred)
    
    # MSLE (primary metric)
    metrics[f"{prefix}msle"] = msle(y_true, y_pred)
    
    # RMSE (both scales)
    metrics[f"{prefix}rmse"] = rmse(y_true, y_pred, log_scale=False)
    metrics[f"{prefix}rmse_log"] = rmse(y_true, y_pred, log_scale=True)
    
    # MAE (both scales)
    metrics[f"{prefix}mae"] = mae(y_true, y_pred, log_scale=False)
    metrics[f"{prefix}mae_log"] = mae(y_true, y_pred, log_scale=True)
    
    # Mean predictions
    metrics[f"{prefix}mean_true"] = float(np.mean(y_true))
    metrics[f"{prefix}mean_pred"] = float(np.mean(y_pred))
    
    # Percentage of zeros
    metrics[f"{prefix}zero_rate_true"] = float(np.mean(y_true == 0))
    metrics[f"{prefix}zero_rate_pred"] = float(np.mean(y_pred == 0))
    
    return metrics


def evaluate_two_stage(
    y_true_revenue: np.ndarray,
    y_true_buyer: np.ndarray,
    p_buyer: np.ndarray,
    y_pred_revenue: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """Comprehensive evaluation for two-stage revenue prediction.
    
    Stage 1: Buyer classification
    Stage 2: Revenue regression
    Final: p(buyer) × E[revenue]
    
    Args:
        y_true_revenue: Ground truth revenue (original scale)
        y_true_buyer: Ground truth buyer labels (0/1)
        p_buyer: Predicted buyer probabilities
        y_pred_revenue: Predicted revenue from regressor
        prefix: Prefix for metric names
        
    Returns:
        Dict of metrics for both stages and final prediction
        
    Example:
        >>> from src.utils.metrics import evaluate_two_stage
        >>> final_pred = p_buyer * rev_pred
        >>> metrics = evaluate_two_stage(
        ...     rev_val, buyer_val, p_buyer_val, rev_pred_val, prefix="val_"
        ... )
    """
    metrics = {}
    
    # Stage 1: Classifier metrics
    cls_metrics = evaluate_classifier(y_true_buyer, p_buyer, prefix=f"{prefix}cls_")
    metrics.update(cls_metrics)
    
    # Stage 2: Regressor metrics (on positive examples only for fair comparison)
    # But for final metric, we need to evaluate on all examples
    
    # Final two-stage prediction
    y_pred_final = p_buyer * y_pred_revenue
    final_metrics = evaluate_regressor(y_true_revenue, y_pred_final, prefix=f"{prefix}final_")
    metrics.update(final_metrics)
    
    # Also evaluate regressor alone (for debugging)
    reg_metrics = evaluate_regressor(y_true_revenue, y_pred_revenue, prefix=f"{prefix}reg_")
    metrics.update(reg_metrics)
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """Pretty print metrics.
    
    Args:
        metrics: Dict of metric names to values
        title: Title to display
        
    Example:
        >>> from src.utils.metrics import evaluate_classifier, print_metrics
        >>> metrics = evaluate_classifier(y_val, p_pred_val, prefix="val_")
        >>> print_metrics(metrics, "Validation Results")
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")
    
    # Group metrics by prefix
    prefixes = set(k.split("_")[0] for k in metrics.keys() if "_" in k)
    
    for prefix in sorted(prefixes):
        prefix_metrics = {k: v for k, v in metrics.items() if k.startswith(prefix)}
        if prefix_metrics:
            print(f"\n{prefix.upper()}:")
            for name, value in sorted(prefix_metrics.items()):
                print(f"  {name:30s}: {value:10.6f}")
    
    # Metrics without prefix
    other_metrics = {k: v for k, v in metrics.items() if "_" not in k}
    if other_metrics:
        print(f"\nOTHER:")
        for name, value in sorted(other_metrics.items()):
            print(f"  {name:30s}: {value:10.6f}")
    
    print(f"{'=' * 60}\n")


def log_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """Log metrics using logger.
    
    Args:
        metrics: Dict of metric names to values
        title: Title for logging
        
    Example:
        >>> from src.utils.metrics import evaluate_classifier, log_metrics
        >>> metrics = evaluate_classifier(y_val, p_pred_val, prefix="val_")
        >>> log_metrics(metrics, "Validation Results")
    """
    LOGGER.info("=" * 60)
    LOGGER.info("%s", title)
    LOGGER.info("=" * 60)
    
    for name, value in sorted(metrics.items()):
        LOGGER.info("  %s: %.6f", name, value)
    
    LOGGER.info("=" * 60)



