"""Fast inference pipeline for revenue prediction.

This module provides a RevenuePredictor class that:
1. Loads student models and artifacts
2. Builds online features from raw data
3. Predicts revenue using two-stage approach
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import dask.dataframe as dd
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.lookup_tables import load_lookup_tables
from src.features.online import build_online_features, load_encoders
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class RevenuePredictor:
    """Fast two-stage revenue predictor using student models.
    
    This class encapsulates the full inference pipeline:
    - Feature engineering (online features only)
    - Two-stage prediction (p(buyer) × E[revenue])
    - Fast enough for production deployment
    
    Example:
        >>> from src.inference.predictor import RevenuePredictor
        >>> 
        >>> # Initialize predictor
        >>> predictor = RevenuePredictor()
        >>> 
        >>> # Load test data
        >>> import pandas as pd
        >>> df_test = pd.read_parquet("data/raw/test/test")
        >>> 
        >>> # Predict
        >>> predictions = predictor.predict(df_test)
        >>> 
        >>> # Create submission
        >>> submission = pd.DataFrame({
        ...     "row_id": df_test["row_id"],
        ...     "iap_revenue_d7": predictions
        ... })
    """
    
    def __init__(
        self,
        project_root: str | Path | None = None,
        student_cls_path: str | Path | None = None,
        student_reg_path: str | Path | None = None,
        encoders_path: str | Path | None = None,
        lookup_tables_path: str | Path | None = None
    ):
        """Initialize predictor with models and artifacts.
        
        Args:
            project_root: Root directory of project (auto-detects if None)
            student_cls_path: Path to student classifier model
            student_reg_path: Path to student regressor model
            encoders_path: Path to encoders pickle
            lookup_tables_path: Path to lookup tables directory
        """
        # Auto-detect project root
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        else:
            project_root = Path(project_root)
        
        # Default paths
        if student_cls_path is None:
            student_cls_path = project_root / "models" / "students" / "student_classifier_lgb.txt"
        if student_reg_path is None:
            student_reg_path = project_root / "models" / "students" / "student_regressor_lgb.txt"
        if encoders_path is None:
            encoders_path = project_root / "data" / "processed" / "encoders" / "online_encoders.pkl"
        if lookup_tables_path is None:
            lookup_tables_path = project_root / "data" / "processed" / "lookup_tables"
        
        LOGGER.info("Loading models and artifacts...")
        
        # Load models
        LOGGER.info(f"Loading classifier from {student_cls_path}")
        self.classifier = lgb.Booster(model_file=str(student_cls_path))
        
        LOGGER.info(f"Loading regressor from {student_reg_path}")
        self.regressor = lgb.Booster(model_file=str(student_reg_path))
        
        # Load encoders
        LOGGER.info(f"Loading encoders from {encoders_path}")
        self.encoders = load_encoders(encoders_path)
        
        # Load lookup tables
        LOGGER.info(f"Loading lookup tables from {lookup_tables_path}")
        self.lookup_tables = load_lookup_tables(lookup_tables_path)
        
        LOGGER.info("✅ Predictor initialized")
    
    def _make_features(self, df: pd.DataFrame | dd.DataFrame) -> pd.DataFrame:
        """Build online features from raw data.
        
        Args:
            df: Raw data (pandas or Dask DataFrame)
            
        Returns:
            Feature DataFrame ready for prediction
        """
        LOGGER.info("Building online features...")
        
        # Convert pandas to Dask if needed
        if isinstance(df, pd.DataFrame):
            df_dask = dd.from_pandas(df, npartitions=4)
        else:
            df_dask = df
        
        # Build features
        X_dask, _ = build_online_features(
            df_dask,
            lookup_tables=self.lookup_tables,
            encoders=self.encoders,
            fit_encoders=False
        )
        
        # Compute if Dask
        if isinstance(X_dask, dd.DataFrame):
            X = X_dask.compute()
        else:
            X = X_dask
        
        LOGGER.info(f"Features built: {X.shape}")
        
        return X
    
    def predict(self, df: pd.DataFrame | dd.DataFrame) -> np.ndarray:
        """Predict revenue for batch of rows.
        
        Two-stage prediction:
        1. Predict p(buyer) using classifier
        2. Predict E[log(revenue)] using regressor
        3. Final: p(buyer) × exp(E[log(revenue)])
        
        Args:
            df: Raw test data (same schema as train)
            
        Returns:
            revenue_pred: Predicted iap_revenue_d7 (original scale)
            
        Example:
            >>> predictor = RevenuePredictor()
            >>> predictions = predictor.predict(df_test)
        """
        LOGGER.info(f"Predicting revenue for {len(df):,} rows...")
        
        # Build features
        X = self._make_features(df)
        
        # Stage 1: Predict p(buyer)
        LOGGER.info("Stage 1: Predicting buyer probability...")
        p_buyer = self.classifier.predict(X)
        p_buyer = np.clip(p_buyer, 0.0, 1.0)
        
        LOGGER.info(f"  Predicted buyer rate: {p_buyer.mean():.4f}")
        
        # Stage 2: Predict log(revenue)
        LOGGER.info("Stage 2: Predicting revenue...")
        log_rev = self.regressor.predict(X)
        rev = np.maximum(0.0, np.expm1(log_rev))
        
        LOGGER.info(f"  Mean predicted revenue: ${rev.mean():.2f}")
        
        # Combine (two-stage)
        revenue_pred = p_buyer * rev
        
        LOGGER.info(f"✅ Final predicted revenue (2-stage): ${revenue_pred.mean():.2f}")
        
        return revenue_pred
    
    def predict_proba_buyer(self, df: pd.DataFrame | dd.DataFrame) -> np.ndarray:
        """Predict only buyer probability (Stage 1).
        
        Args:
            df: Raw data
            
        Returns:
            p_buyer: Predicted buyer probabilities
        """
        X = self._make_features(df)
        p_buyer = self.classifier.predict(X)
        return np.clip(p_buyer, 0.0, 1.0)
    
    def predict_revenue_given_buyer(self, df: pd.DataFrame | dd.DataFrame) -> np.ndarray:
        """Predict only revenue (Stage 2, not conditioned on buyer).
        
        Args:
            df: Raw data
            
        Returns:
            revenue_pred: Predicted revenue (original scale)
        """
        X = self._make_features(df)
        log_rev = self.regressor.predict(X)
        return np.maximum(0.0, np.expm1(log_rev))



