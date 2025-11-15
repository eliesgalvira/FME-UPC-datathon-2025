"""Fast online features for student models.

Constraints:
- < 100μs per prediction
- No groupby, no histogram rebuilds, no heavy aggregations
- Simple scalars, lookups, sum/len of lists only

Features built:
1. Request features: categorical + temporal
2. User scalar features: activity + ratios
3. Simple aggregates from complex columns (lists/dicts)
4. Precomputed lookup tables (O(1) dict lookups)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


# Feature column definitions
CATEGORICAL_FEATURES = [
    "country",
    "region",
    "dev_os",
    "dev_osv",
    "dev_make",
    "dev_model",
    "carrier",
    "advertiser_bundle",
    "advertiser_category",
    "advertiser_subcategory",
    "advertiser_bottom_taxonomy_level",
]

USER_SCALAR_FEATURES = [
    "avg_act_days",
    "avg_daily_sessions",
    "avg_days_ins",
    "avg_duration",
    "weekend_ratio",
    "weeks_since_first_seen",
    "wifi_ratio",
]

TEMPORAL_FEATURES = [
    "hour",
    "release_date",
]


def _extract_list_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract simple aggregates from list/array columns.
    
    Pure function: takes DataFrame partition, returns new DataFrame with added features.
    
    Args:
        df: Input DataFrame partition
        
    Returns:
        DataFrame with added list aggregate features
    """
    result = df.copy()
    
    # From user_bundles: number of installed apps
    if "user_bundles" in df.columns:
        result["num_user_bundles"] = df["user_bundles"].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) and x is not None else 0
        )
    
    # From user_bundles_l28d: recent installed apps
    if "user_bundles_l28d" in df.columns:
        result["num_user_bundles_l28d"] = df["user_bundles_l28d"].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) and x is not None else 0
        )
    
    # From bundles_ins: number of bundles with installs
    if "bundles_ins" in df.columns:
        result["num_bundles_ins"] = df["bundles_ins"].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) and x is not None else 0
        )
    
    # From new_bundles: number of new bundles
    if "new_bundles" in df.columns:
        result["num_new_bundles"] = df["new_bundles"].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) and x is not None else 0
        )
    
    return result


def _extract_dict_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract simple aggregates from dict/map columns.
    
    Pure function: takes DataFrame partition, returns new DataFrame with added features.
    
    Args:
        df: Input DataFrame partition
        
    Returns:
        DataFrame with added dict aggregate features
    """
    result = df.copy()
    
    # From num_buys_bundle: total past buys
    if "num_buys_bundle" in df.columns:
        result["total_buys"] = df["num_buys_bundle"].apply(
            lambda x: sum(x.values()) if isinstance(x, dict) and x else 0
        )
        result["num_bought_bundles"] = df["num_buys_bundle"].apply(
            lambda x: len(x) if isinstance(x, dict) and x else 0
        )
    
    # From iap_revenue_usd_bundle: total past IAP revenue
    if "iap_revenue_usd_bundle" in df.columns:
        result["total_past_revenue"] = df["iap_revenue_usd_bundle"].apply(
            lambda x: sum(x.values()) if isinstance(x, dict) and x else 0.0
        )
        result["num_revenue_bundles"] = df["iap_revenue_usd_bundle"].apply(
            lambda x: len(x) if isinstance(x, dict) and x else 0
        )
    
    # From iap_revenue_usd_category: revenue by category
    if "iap_revenue_usd_category" in df.columns:
        result["total_category_revenue"] = df["iap_revenue_usd_category"].apply(
            lambda x: sum(x.values()) if isinstance(x, dict) and x else 0.0
        )
    
    return result


def _encode_hour_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour as sine/cosine for cyclical nature.
    
    Insight from EDA: Hour is highly predictive (revenue spikes at 18h, 00-02h, 16h).
    Cyclical encoding captures that 23h and 0h are close.
    
    Args:
        df: Input DataFrame with 'hour' column
        
    Returns:
        DataFrame with hour_sin and hour_cos features
    """
    result = df.copy()
    
    if "hour" in df.columns:
        # Parse hour string (format: "HH" or similar)
        # Handle both string and numeric hour
        hour_numeric = pd.to_numeric(result["hour"], errors="coerce").fillna(0)
        
        # Encode as sin/cos (24-hour cycle)
        result["hour_sin"] = np.sin(2 * np.pi * hour_numeric / 24)
        result["hour_cos"] = np.cos(2 * np.pi * hour_numeric / 24)
        
        # Also keep hour bins (morning, afternoon, evening, night)
        result["hour_bin"] = pd.cut(
            hour_numeric,
            bins=[0, 6, 12, 18, 24],
            labels=[0, 1, 2, 3],  # night, morning, afternoon, evening
            include_lowest=True
        ).astype("int8")
    
    return result


def _add_lookup_features(
    df: pd.DataFrame,
    lookup_tables: Dict[str, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:
    """Add precomputed lookup table features.
    
    O(1) dict lookups for each row - very fast!
    
    Args:
        df: Input DataFrame
        lookup_tables: Dict with keys ['bundle', 'category', 'country', 'segment']
        
    Returns:
        DataFrame with added lookup features
    """
    result = df.copy()
    
    # Bundle stats
    if "bundle" in lookup_tables and "advertiser_bundle" in df.columns:
        bundle_stats = lookup_tables["bundle"]
        result["bundle_buyer_rate"] = df["advertiser_bundle"].map(
            lambda x: bundle_stats.get(x, {}).get("buyer_rate", 0.0)
        )
        result["bundle_avg_rev_d7"] = df["advertiser_bundle"].map(
            lambda x: bundle_stats.get(x, {}).get("avg_rev_d7", 0.0)
        )
        result["bundle_whale_rate"] = df["advertiser_bundle"].map(
            lambda x: bundle_stats.get(x, {}).get("whale_rate", 0.0)
        )
    
    # Category stats
    if "category" in lookup_tables and "advertiser_category" in df.columns:
        cat_stats = lookup_tables["category"]
        result["category_buyer_rate"] = df["advertiser_category"].map(
            lambda x: cat_stats.get(x, {}).get("buyer_rate", 0.0)
        )
        result["category_avg_rev_d7"] = df["advertiser_category"].map(
            lambda x: cat_stats.get(x, {}).get("avg_rev_d7", 0.0)
        )
        result["category_whale_rate"] = df["advertiser_category"].map(
            lambda x: cat_stats.get(x, {}).get("whale_rate", 0.0)
        )
    
    # Country stats
    if "country" in lookup_tables and "country" in df.columns:
        country_stats = lookup_tables["country"]
        result["country_buyer_rate"] = df["country"].map(
            lambda x: country_stats.get(x, {}).get("buyer_rate", 0.0)
        )
        result["country_avg_rev_d7"] = df["country"].map(
            lambda x: country_stats.get(x, {}).get("avg_rev_d7", 0.0)
        )
        result["country_whale_rate"] = df["country"].map(
            lambda x: country_stats.get(x, {}).get("whale_rate", 0.0)
        )
    
    # Segment stats (country, dev_os)
    if "segment" in lookup_tables and "country" in df.columns and "dev_os" in df.columns:
        segment_stats = lookup_tables["segment"]
        result["segment_buyer_rate"] = df.apply(
            lambda row: segment_stats.get(
                f"{row['country']}_{row['dev_os']}", {}
            ).get("buyer_rate", 0.0),
            axis=1
        )
        result["segment_avg_rev_d7"] = df.apply(
            lambda row: segment_stats.get(
                f"{row['country']}_{row['dev_os']}", {}
            ).get("avg_rev_d7", 0.0),
            axis=1
        )
    
    return result


def build_online_features(
    ddf: dd.DataFrame,
    lookup_tables: Optional[Dict[str, Dict]] = None,
    encoders: Optional[Dict[str, Any]] = None,
    fit_encoders: bool = True
) -> Tuple[dd.DataFrame, Dict[str, Any]]:
    """Build fast online features for student models.
    
    This function extracts cheap features that can be computed in < 100μs per prediction.
    Uses Dask best practices: map_partitions for custom operations, avoid compute().
    
    Args:
        ddf: Input Dask DataFrame
        lookup_tables: Optional precomputed statistics (if None, features are 0)
        encoders: Optional pre-fitted encoders (for inference)
        fit_encoders: If True and encoders is None, fit new encoders
        
    Returns:
        X_online: Dask DataFrame with encoded features
        encoders: Dict of fitted encoders (save for inference)
        
    Example:
        >>> from src.data.loader import DataLoader
        >>> from src.features.online import build_online_features
        >>> 
        >>> loader = DataLoader(config={...})
        >>> ddf_train, ddf_val = loader.load_train()
        >>> 
        >>> # Training: fit encoders
        >>> X_train, encoders = build_online_features(ddf_train, lookup_tables)
        >>> 
        >>> # Validation: reuse encoders
        >>> X_val, _ = build_online_features(ddf_val, lookup_tables, encoders=encoders, fit_encoders=False)
    """
    LOGGER.info("Building online features (fast features for students)...")
    
    # Start with copy to avoid modifying input
    result = ddf.copy()
    
    # 1. Extract simple aggregates from complex columns
    LOGGER.info("Extracting list/dict aggregates...")
    # Note: Don't specify meta, let Dask infer it from the first partition
    result = result.map_partitions(_extract_list_features)
    result = result.map_partitions(_extract_dict_features)
    
    # 2. Encode hour cyclically
    LOGGER.info("Encoding hour cyclically (sin/cos)...")
    result = result.map_partitions(_encode_hour_cyclical)
    
    # 3. Add lookup table features (if provided)
    if lookup_tables is not None:
        LOGGER.info("Adding lookup table features...")
        result = result.map_partitions(
            _add_lookup_features,
            lookup_tables=lookup_tables
        )
    
    # 4. Select feature columns
    feature_cols = []
    
    # Categorical features (will be encoded)
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in result.columns]
    feature_cols.extend(categorical_cols)
    
    # User scalar features
    scalar_cols = [c for c in USER_SCALAR_FEATURES if c in result.columns]
    feature_cols.extend(scalar_cols)
    
    # Temporal features (hour already encoded, but keep for now)
    temporal_cols = [c for c in TEMPORAL_FEATURES if c in result.columns]
    feature_cols.extend(temporal_cols)
    
    # Derived features (from list/dict/hour encoding)
    derived_cols = [
        "num_user_bundles", "num_user_bundles_l28d", "num_bundles_ins", "num_new_bundles",
        "total_buys", "num_bought_bundles", "total_past_revenue", "num_revenue_bundles",
        "total_category_revenue", "hour_sin", "hour_cos", "hour_bin"
    ]
    derived_cols = [c for c in derived_cols if c in result.columns]
    feature_cols.extend(derived_cols)
    
    # Lookup features
    if lookup_tables is not None:
        lookup_cols = [
            c for c in result.columns
            if c.endswith(("_buyer_rate", "_avg_rev_d7", "_whale_rate"))
        ]
        feature_cols.extend(lookup_cols)
    
    # Select only feature columns
    result = result[feature_cols]
    
    # 5. Encode categorical features
    if encoders is None and fit_encoders:
        LOGGER.info("Fitting ordinal encoders for categorical features...")
        encoders = {}
        
        # Sample first few partitions to get unique values (more memory efficient)
        # This avoids computing the entire dataset
        LOGGER.info("Sampling data to fit encoders (using first 3 partitions)...")
        n_sample_partitions = min(3, result.npartitions)
        sample_df = result.head(n=100000, npartitions=n_sample_partitions, compute=True)
        
        # Fit encoders on sample
        for col in categorical_cols:
            if col in sample_df.columns:
                LOGGER.debug(f"Fitting encoder for {col}...")
                # Get unique values from sample
                unique_vals = sample_df[col].dropna().unique()
                
                if len(unique_vals) == 0:
                    LOGGER.warning(f"No non-null values for {col}, skipping encoder")
                    continue
                
                # Create encoder
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.int32
                )
                encoder.fit(unique_vals.reshape(-1, 1))
                encoders[col] = encoder
        
        LOGGER.info(f"Fitted {len(encoders)} encoders from sample data")
    
    # Apply encoders
    if encoders:
        LOGGER.info("Applying encoders to categorical features...")
        
        def apply_encoders_partition(df: pd.DataFrame, encoders: Dict) -> pd.DataFrame:
            """Apply encoders to a partition."""
            result = df.copy()
            for col, encoder in encoders.items():
                if col in df.columns:
                    # Handle missing values
                    mask = pd.isna(df[col])
                    result[col] = encoder.transform(df[col].values.reshape(-1, 1)).flatten()
                    result.loc[mask, col] = -1  # Use -1 for missing
            return result
        
        result = result.map_partitions(
            apply_encoders_partition,
            encoders=encoders
        )
    
    # 6. Fill missing values
    LOGGER.info("Filling missing values...")
    # Categorical: already handled (-1)
    # Numeric: fill with 0
    numeric_cols = scalar_cols + derived_cols
    if lookup_tables:
        numeric_cols += [c for c in feature_cols if c.endswith(("_buyer_rate", "_avg_rev_d7", "_whale_rate"))]
    numeric_cols = [c for c in numeric_cols if c in result.columns]
    
    result[numeric_cols] = result[numeric_cols].fillna(0.0)
    
    LOGGER.info(f"✅ Online features built: {len(result.columns)} features")
    
    return result, encoders or {}


def load_lookup_tables(path: str | Path) -> Dict[str, Dict]:
    """Load precomputed lookup tables from disk.
    
    Args:
        path: Path to directory containing lookup table JSON files
        
    Returns:
        Dict with keys ['bundle', 'category', 'country', 'segment']
        
    Example:
        >>> lookup_tables = load_lookup_tables("data/processed/lookup_tables")
        >>> X, encoders = build_online_features(ddf, lookup_tables)
    """
    path = Path(path)
    tables = {}
    
    for name in ["bundle", "category", "country", "segment"]:
        file_path = path / f"{name}_stats.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                tables[name] = json.load(f)
            LOGGER.info(f"Loaded {name} lookup table: {len(tables[name])} entries")
        else:
            LOGGER.warning(f"Lookup table not found: {file_path}")
            tables[name] = {}
    
    return tables


def save_encoders(encoders: Dict[str, Any], path: str | Path) -> None:
    """Save encoders to disk for inference.
    
    Args:
        encoders: Dict of fitted encoders
        path: Path to save pickle file
        
    Example:
        >>> X, encoders = build_online_features(ddf_train, lookup_tables)
        >>> save_encoders(encoders, "data/processed/encoders/online_encoders.pkl")
    """
    import pickle
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(encoders, f)
    
    LOGGER.info(f"✅ Saved {len(encoders)} encoders to {path}")


def load_encoders(path: str | Path) -> Dict[str, Any]:
    """Load encoders from disk for inference.
    
    Args:
        path: Path to pickle file
        
    Returns:
        Dict of encoders
        
    Example:
        >>> encoders = load_encoders("data/processed/encoders/online_encoders.pkl")
        >>> X, _ = build_online_features(ddf_test, lookup_tables, encoders=encoders, fit_encoders=False)
    """
    import pickle
    
    with open(path, "rb") as f:
        encoders = pickle.load(f)
    
    LOGGER.info(f"✅ Loaded {len(encoders)} encoders from {path}")
    
    return encoders

