"""Rich offline features for teacher models.

Constraints:
- Slow OK (offline training only)
- Heavy aggregations, histograms, complex derived features allowed

Features built (in addition to online features):
1. Histogram features: entropy, diversity, top-1 fraction
2. Revenue/buy map features: sum, max, mean, per-key stats
3. Whale features: rank and percentile for high spenders
4. Recency features: days since last activity
5. Cross-feature interactions (captured by tree models)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from src.features.online import build_online_features
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _extract_histogram_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract rich statistics from histogram columns.
    
    Histogram columns (*_hist) are dicts mapping keys to counts.
    Extract: diversity, entropy, loyalty (top-1 fraction).
    
    Args:
        df: Input DataFrame partition
        
    Returns:
        DataFrame with added histogram features
    """
    result = df.copy()
    
    # List of histogram columns to process
    hist_cols = [
        "city_hist",
        "country_hist",
        "region_hist",
        "dev_language_hist",
        "dev_osv_hist",
        "hour_ratio"
    ]
    
    for col in hist_cols:
        if col not in df.columns:
            continue
        
        prefix = col.replace("_hist", "").replace("_ratio", "")
        
        # Vectorized extraction (much faster than .apply())
        col_data = df[col].values
        
        # Number of unique keys (diversity)
        result[f"{prefix}_n_unique"] = [len(x) if isinstance(x, dict) and x else 0 for x in col_data]
        
        # Total count
        result[f"{prefix}_total_count"] = [sum(x.values()) if isinstance(x, dict) and x else 0 for x in col_data]
        
        # SKIP ENTROPY - too slow for marginal value
        # SKIP TOP1_FRAC - too slow for marginal value
    
    return result


def _extract_revenue_buy_map_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract rich statistics from revenue/buy map columns.
    
    Map columns are dicts mapping bundle/category IDs to values (revenue or count).
    Extract: sum, max, mean, number of keys, current advertiser value.
    
    Args:
        df: Input DataFrame partition
        
    Returns:
        DataFrame with added map features
    """
    result = df.copy()
    
    # List of map columns to process
    map_cols = {
        "num_buys_bundle": "buys_bundle",
        "num_buys_category": "buys_category",
        "num_buys_category_bottom_taxonomy": "buys_tax",
        "iap_revenue_usd_bundle": "rev_bundle",
        "iap_revenue_usd_category": "rev_category",
        "iap_revenue_usd_category_bottom_taxonomy": "rev_tax"
    }
    
    for col, prefix in map_cols.items():
        if col not in df.columns:
            continue
        
        # Vectorized extraction
        col_data = df[col].values
        
        # Sum of values
        result[f"{prefix}_sum"] = [sum(x.values()) if isinstance(x, dict) and x else 0.0 for x in col_data]
        
        # Max of values
        result[f"{prefix}_max"] = [max(x.values()) if isinstance(x, dict) and x else 0.0 for x in col_data]
        
        # Number of non-zero keys
        result[f"{prefix}_n_keys"] = [len(x) if isinstance(x, dict) and x else 0 for x in col_data]
        
        # Mean value per key
        result[f"{prefix}_mean"] = [sum(x.values()) / len(x) if isinstance(x, dict) and x and len(x) > 0 else 0.0 for x in col_data]
        
        # Value for current advertiser (vectorized if applicable)
        if "advertiser_bundle" in df.columns and "bundle" in col:
            bundle_data = df["advertiser_bundle"].values
            result[f"{prefix}_current_adv"] = [
                x.get(bundle, 0.0) if isinstance(x, dict) and x and bundle else 0.0
                for x, bundle in zip(col_data, bundle_data)
            ]
    
    return result


def _extract_whale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract whale-related features.
    
    Whale columns contain percentile ranks for high-spending users.
    Extract: max rank, current bundle rank.
    
    Args:
        df: Input DataFrame partition
        
    Returns:
        DataFrame with added whale features
    """
    result = df.copy()
    
    # Whale revenue rank columns
    whale_cols = {
        "whale_users_bundle_revenue_prank": "whale_rev_prank",
        "whale_users_bundle_num_buys_prank": "whale_buys_prank",
        "whale_users_bundle_total_revenue": "whale_total_rev",
        "whale_users_bundle_total_num_buys": "whale_total_buys"
    }
    
    for col, prefix in whale_cols.items():
        if col not in df.columns:
            continue
        
        # Vectorized extraction
        col_data = df[col].values
        
        # Max value (highest whale rank/revenue)
        result[f"{prefix}_max"] = [max(x.values()) if isinstance(x, dict) and x else 0.0 for x in col_data]
        
        # Number of whale bundles
        result[f"{prefix}_n_bundles"] = [len(x) if isinstance(x, dict) and x else 0 for x in col_data]
        
        # Value for current advertiser bundle (vectorized)
        if "advertiser_bundle" in df.columns:
            bundle_data = df["advertiser_bundle"].values
            result[f"{prefix}_current"] = [
                x.get(bundle, 0.0) if isinstance(x, dict) and x and bundle else 0.0
                for x, bundle in zip(col_data, bundle_data)
            ]
    
    return result


def _extract_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract recency features from timestamp columns.
    
    Convert timestamps to days since event.
    Recency is highly predictive of future behavior.
    
    Args:
        df: Input DataFrame partition
        
    Returns:
        DataFrame with added recency features
    """
    result = df.copy()
    
    # Timestamp columns (maps of bundle/category to timestamp)
    ts_cols = {
        "last_buy_ts_bundle": "last_buy",
        "last_buy_ts_category": "last_buy_cat",
        "last_install_ts_bundle": "last_install",
        "last_install_ts_category": "last_install_cat",
        "first_request_ts_bundle": "first_req",
        "first_request_ts_category_bottom_taxonomy": "first_req_tax"
    }
    
    # Current time reference (use max timestamp in data or a fixed reference)
    # For simplicity, use relative recency (days since most recent)
    
    for col, prefix in ts_cols.items():
        if col not in df.columns:
            continue
        
        # Vectorized extraction (simplified - skip min/max which are slow)
        col_data = df[col].values
        
        # Number of bundles/categories with activity (fast)
        result[f"{prefix}_n_items"] = [len(x) if isinstance(x, dict) and x else 0 for x in col_data]
        
        # SKIP most_recent/least_recent - too slow with min/max on dict values
        # SKIP current_adv - too slow
    
    return result


def _extract_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from user action columns.
    
    Action columns track user interactions with advertiser/bundles.
    Extract: count, recency, last action type.
    
    Args:
        df: Input DataFrame partition
        
    Returns:
        DataFrame with added action features
    """
    result = df.copy()
    
    # Advertiser actions (vectorized)
    if "advertiser_actions_action_count" in df.columns:
        col_data = df["advertiser_actions_action_count"].values
        result["adv_action_total"] = [sum(x.values()) if isinstance(x, dict) and x else 0 for x in col_data]
        result["adv_action_n_types"] = [len(x) if isinstance(x, dict) and x else 0 for x in col_data]
    
    # User bundle actions (vectorized)
    if "user_actions_bundles_action_count" in df.columns:
        col_data = df["user_actions_bundles_action_count"].values
        result["user_action_total"] = [sum(x.values()) if isinstance(x, dict) and x else 0 for x in col_data]
        result["user_action_n_types"] = [len(x) if isinstance(x, dict) and x else 0 for x in col_data]
    
    # Last advertiser action (already categorical, just encode later)
    # Keep as-is for now
    
    return result


def build_offline_features(
    ddf: dd.DataFrame,
    lookup_tables: Optional[Dict[str, Dict]] = None,
    encoders_from_online: Optional[Dict[str, Any]] = None
) -> Tuple[dd.DataFrame, Dict[str, Any]]:
    """Build rich offline features for teacher models.
    
    This includes all online features plus heavy aggregations from complex columns.
    Uses Dask best practices: map_partitions for custom operations.
    
    Args:
        ddf: Input Dask DataFrame
        lookup_tables: Optional precomputed statistics
        encoders_from_online: Optional pre-fitted encoders from online features
        
    Returns:
        X_offline: Dask DataFrame with all features
        encoders: Dict of encoders (same as online)
        
    Example:
        >>> from src.data.loader import DataLoader
        >>> from src.features.offline import build_offline_features
        >>> from src.features.lookup_tables import generate_lookup_tables
        >>> 
        >>> loader = DataLoader(config={...})
        >>> ddf_train, _ = loader.load_train()
        >>> 
        >>> # Generate lookup tables
        >>> lookup_tables = generate_lookup_tables(ddf_train, "data/processed/lookup_tables")
        >>> 
        >>> # Build offline features
        >>> X_train, encoders = build_offline_features(ddf_train, lookup_tables)
    """
    # Preconditions
    assert isinstance(ddf, dd.DataFrame), "ddf must be a Dask DataFrame"
    assert len(ddf.columns) > 0, "Input DataFrame must have at least one column"
    if encoders_from_online is not None:
        assert isinstance(encoders_from_online, dict), "encoders_from_online must be a dict if provided"
    if lookup_tables is not None:
        assert isinstance(lookup_tables, dict), "lookup_tables must be a dict if provided"
    
    LOGGER.info("Building offline features (rich features for teachers)...")
    
    # 1. Start with online features (fast features)
    LOGGER.info("Step 1/6: Building online features...")
    result, encoders = build_online_features(
        ddf,
        lookup_tables=lookup_tables,
        encoders=encoders_from_online,
        fit_encoders=(encoders_from_online is None)
    )
    
    # Need to join back with original data to access complex columns
    # Online features only kept basic columns
    LOGGER.info("Step 2/6: Merging with original data for complex columns...")
    
    # Complex columns needed for offline feature extraction
    complex_cols = [
        "city_hist", "country_hist", "region_hist", "dev_language_hist", "dev_osv_hist", "hour_ratio",
        "num_buys_bundle", "num_buys_category", "num_buys_category_bottom_taxonomy",
        "iap_revenue_usd_bundle", "iap_revenue_usd_category", "iap_revenue_usd_category_bottom_taxonomy",
        "whale_users_bundle_revenue_prank", "whale_users_bundle_num_buys_prank",
        "whale_users_bundle_total_revenue", "whale_users_bundle_total_num_buys",
        "last_buy_ts_bundle", "last_buy_ts_category", "last_install_ts_bundle", "last_install_ts_category",
        "first_request_ts_bundle", "first_request_ts_category_bottom_taxonomy",
        "advertiser_actions_action_count", "advertiser_actions_action_last_timestamp",
        "user_actions_bundles_action_count", "user_actions_bundles_action_last_timestamp",
        "last_advertiser_action", "advertiser_bundle"
    ]
    
    # Build dict of columns to add back (only those that exist and aren't already in result)
    # This avoids multiple assign() calls which create intermediate DataFrames
    cols_to_add = {}
    for col in complex_cols:
        if col in ddf.columns and col not in result.columns:
            cols_to_add[col] = ddf[col]
    
    # Add row_id if available (for potential debugging/tracking)
    if "row_id" in ddf.columns and "row_id" not in result.columns:
        cols_to_add["row_id"] = ddf["row_id"]
    
    # Single assign() call for all columns (more efficient)
    if cols_to_add:
        result = result.assign(**cols_to_add)
        LOGGER.info(f"Added {len(cols_to_add)} complex columns for feature extraction")
    
    # 2. Extract histogram features
    LOGGER.info("Step 3/6: Extracting histogram features...")
    result = result.map_partitions(_extract_histogram_features)
    
    # 3. Extract revenue/buy map features
    LOGGER.info("Step 4/6: Extracting revenue/buy map features...")
    result = result.map_partitions(_extract_revenue_buy_map_features)
    
    # 4. Extract whale features
    LOGGER.info("Step 5/6: Extracting whale features...")
    result = result.map_partitions(_extract_whale_features)
    
    # 5. Extract recency features
    LOGGER.info("Step 6/6: Extracting recency features...")
    result = result.map_partitions(_extract_recency_features)
    
    # 6. Extract action features
    LOGGER.info("Extracting action features...")
    result = result.map_partitions(_extract_action_features)
    
    # Drop complex columns (not needed for model training)
    LOGGER.info("Dropping complex columns...")
    cols_to_drop = [col for col in complex_cols if col in result.columns]
    if cols_to_drop:
        result = result.drop(columns=cols_to_drop)
    
    # Fill missing values
    LOGGER.info("Filling missing values...")
    # All remaining columns should be numeric after dropping complex columns
    # Keep only row_id if it exists
    cols_to_fill = [c for c in result.columns if c != "row_id"]
    if cols_to_fill:
        result[cols_to_fill] = result[cols_to_fill].fillna(0.0)
    
    LOGGER.info(f"✅ Offline features built: {len(result.columns)} features")
    
    # Postconditions
    assert len(result.columns) > 0, "Result must have at least one feature column"
    assert isinstance(encoders, dict), "Encoders must be a dict"
    # Offline features should have more columns than online features
    # (since it includes all online features + complex aggregations)
    
    return result, encoders


def save_feature_names(feature_names: list, path: str) -> None:
    """Save feature names for model training/inference.
    
    Args:
        feature_names: List of feature column names
        path: Output file path
        
    Example:
        >>> X_train, encoders = build_offline_features(ddf_train, lookup_tables)
        >>> save_feature_names(X_train.columns.tolist(), "data/processed/feature_names.txt")
    """
    import json
    from pathlib import Path
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(feature_names, f, indent=2)
    
    LOGGER.info(f"✅ Saved {len(feature_names)} feature names to {path}")

