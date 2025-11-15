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
        
        # Number of unique keys (diversity)
        result[f"{prefix}_n_unique"] = df[col].apply(
            lambda x: len(x) if isinstance(x, dict) and x else 0
        )
        
        # Total count
        result[f"{prefix}_total_count"] = df[col].apply(
            lambda x: sum(x.values()) if isinstance(x, dict) and x else 0
        )
        
        # Entropy (uncertainty measure)
        def calc_entropy(x):
            if not isinstance(x, dict) or not x:
                return 0.0
            counts = np.array(list(x.values()), dtype=float)
            if counts.sum() == 0:
                return 0.0
            probs = counts / counts.sum()
            return float(scipy_entropy(probs))
        
        result[f"{prefix}_entropy"] = df[col].apply(calc_entropy)
        
        # Top-1 fraction (loyalty to one value)
        def calc_top1_frac(x):
            if not isinstance(x, dict) or not x:
                return 0.0
            counts = list(x.values())
            return max(counts) / sum(counts) if sum(counts) > 0 else 0.0
        
        result[f"{prefix}_top1_frac"] = df[col].apply(calc_top1_frac)
    
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
        
        # Sum of values
        result[f"{prefix}_sum"] = df[col].apply(
            lambda x: sum(x.values()) if isinstance(x, dict) and x else 0.0
        )
        
        # Max of values
        result[f"{prefix}_max"] = df[col].apply(
            lambda x: max(x.values()) if isinstance(x, dict) and x else 0.0
        )
        
        # Number of non-zero keys
        result[f"{prefix}_n_keys"] = df[col].apply(
            lambda x: len(x) if isinstance(x, dict) and x else 0
        )
        
        # Mean value per key
        result[f"{prefix}_mean"] = df[col].apply(
            lambda x: sum(x.values()) / len(x) if isinstance(x, dict) and x and len(x) > 0 else 0.0
        )
        
        # Value for current advertiser (if applicable)
        if "advertiser_bundle" in df.columns and "bundle" in col:
            def get_advertiser_value(row):
                map_val = row[col]
                bundle = row["advertiser_bundle"]
                if isinstance(map_val, dict) and map_val and bundle:
                    return map_val.get(bundle, 0.0)
                return 0.0
            
            result[f"{prefix}_current_adv"] = df.apply(get_advertiser_value, axis=1)
    
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
        
        # Max value (highest whale rank/revenue)
        result[f"{prefix}_max"] = df[col].apply(
            lambda x: max(x.values()) if isinstance(x, dict) and x else 0.0
        )
        
        # Number of whale bundles
        result[f"{prefix}_n_bundles"] = df[col].apply(
            lambda x: len(x) if isinstance(x, dict) and x else 0
        )
        
        # Value for current advertiser bundle
        if "advertiser_bundle" in df.columns:
            def get_current_whale_value(row):
                whale_val = row[col]
                bundle = row["advertiser_bundle"]
                if isinstance(whale_val, dict) and whale_val and bundle:
                    return whale_val.get(bundle, 0.0)
                return 0.0
            
            result[f"{prefix}_current"] = df.apply(get_current_whale_value, axis=1)
    
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
        
        # Most recent timestamp (min value = most recent)
        result[f"{prefix}_most_recent"] = df[col].apply(
            lambda x: min(x.values()) if isinstance(x, dict) and x else np.nan
        )
        
        # Least recent timestamp (max value = oldest)
        result[f"{prefix}_least_recent"] = df[col].apply(
            lambda x: max(x.values()) if isinstance(x, dict) and x else np.nan
        )
        
        # Number of bundles/categories with activity
        result[f"{prefix}_n_items"] = df[col].apply(
            lambda x: len(x) if isinstance(x, dict) and x else 0
        )
        
        # For bundle-level timestamps, get value for current advertiser
        if "advertiser_bundle" in df.columns and "bundle" in col:
            def get_advertiser_ts(row):
                ts_map = row[col]
                bundle = row["advertiser_bundle"]
                if isinstance(ts_map, dict) and ts_map and bundle:
                    return ts_map.get(bundle, np.nan)
                return np.nan
            
            result[f"{prefix}_current_adv"] = df.apply(get_advertiser_ts, axis=1)
    
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
    
    # Advertiser actions
    if "advertiser_actions_action_count" in df.columns:
        result["adv_action_total"] = df["advertiser_actions_action_count"].apply(
            lambda x: sum(x.values()) if isinstance(x, dict) and x else 0
        )
        result["adv_action_n_types"] = df["advertiser_actions_action_count"].apply(
            lambda x: len(x) if isinstance(x, dict) and x else 0
        )
    
    # User bundle actions
    if "user_actions_bundles_action_count" in df.columns:
        result["user_action_total"] = df["user_actions_bundles_action_count"].apply(
            lambda x: sum(x.values()) if isinstance(x, dict) and x else 0
        )
        result["user_action_n_types"] = df["user_actions_bundles_action_count"].apply(
            lambda x: len(x) if isinstance(x, dict) and x else 0
        )
    
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
    
    # Keep row_id for joining if available
    if "row_id" in ddf.columns:
        result = result.assign(row_id=ddf["row_id"])
    
    # Also need to add back complex columns for feature extraction
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
    
    # Add back columns that exist in original data
    for col in complex_cols:
        if col in ddf.columns and col not in result.columns:
            result = result.assign(**{col: ddf[col]})
    
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
    cols_to_drop = [col for col in complex_cols if col in result.columns and col != "advertiser_bundle"]
    if cols_to_drop:
        result = result.drop(columns=cols_to_drop)
    
    # Fill missing values
    LOGGER.info("Filling missing values...")
    # All numeric, fill with 0
    numeric_cols = [c for c in result.columns if c not in ["row_id"]]
    result[numeric_cols] = result[numeric_cols].fillna(0.0)
    
    LOGGER.info(f"✅ Offline features built: {len(result.columns)} features")
    
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

