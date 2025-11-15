"""Generate precomputed lookup tables for fast inference.

This module computes statistics from training data that can be used as O(1)
lookup features during inference.

Statistics computed:
- Bundle-level: buyer rate, average revenue, whale rate
- Category-level: same as bundle
- Country-level: same as bundle
- Segment-level (country × dev_os): buyer rate, average revenue

These tables enable students to capture teacher knowledge without heavy computation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import dask.dataframe as dd
import numpy as np

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Whale threshold: users with D7 revenue > this are "whales"
WHALE_THRESHOLD = 10.0  # log1p scale, ~$22k in original scale


def generate_lookup_tables(
    ddf_train: dd.DataFrame,
    output_dir: str | Path,
    whale_threshold: float = WHALE_THRESHOLD
) -> Dict[str, Dict]:
    """Generate lookup tables from training data.
    
    Uses Dask groupby for efficient aggregation.
    Follows best practice: build computation graph, compute once at the end.
    
    Args:
        ddf_train: Training data (Dask DataFrame)
        output_dir: Directory to save lookup tables
        whale_threshold: Revenue threshold to define whales (in log1p scale)
        
    Returns:
        Dict of lookup tables with keys ['bundle', 'category', 'country', 'segment']
        
    Example:
        >>> from src.data.loader import DataLoader
        >>> from src.features.lookup_tables import generate_lookup_tables
        >>> 
        >>> loader = DataLoader(config={...})
        >>> ddf_train, _ = loader.load_train()
        >>> 
        >>> lookup_tables = generate_lookup_tables(
        ...     ddf_train,
        ...     "data/processed/lookup_tables"
        ... )
    """
    LOGGER.info("Generating lookup tables from training data...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare target columns
    # Buyer flag
    if "buyer_d7" not in ddf_train.columns:
        LOGGER.warning("buyer_d7 not in training data, computing from iap_revenue_d7")
        ddf_train = ddf_train.assign(buyer_d7=(ddf_train["iap_revenue_d7"] > 0).astype(int))
    
    # Log revenue
    if "log_rev_d7" not in ddf_train.columns:
        ddf_train = ddf_train.assign(log_rev_d7=np.log1p(ddf_train["iap_revenue_d7"]))
    
    # Whale flag
    if "is_whale" not in ddf_train.columns:
        ddf_train = ddf_train.assign(is_whale=(ddf_train["log_rev_d7"] > whale_threshold).astype(int))
    
    # Generate each lookup table
    tables = {}
    
    # 1. Bundle-level statistics
    if "advertiser_bundle" in ddf_train.columns:
        LOGGER.info("Computing bundle-level statistics...")
        tables["bundle"] = _compute_group_stats(
            ddf_train,
            group_col="advertiser_bundle",
            name="bundle"
        )
        _save_lookup_table(tables["bundle"], output_dir / "bundle_stats.json")
    
    # 2. Category-level statistics
    if "advertiser_category" in ddf_train.columns:
        LOGGER.info("Computing category-level statistics...")
        tables["category"] = _compute_group_stats(
            ddf_train,
            group_col="advertiser_category",
            name="category"
        )
        _save_lookup_table(tables["category"], output_dir / "category_stats.json")
    
    # 3. Country-level statistics
    if "country" in ddf_train.columns:
        LOGGER.info("Computing country-level statistics...")
        tables["country"] = _compute_group_stats(
            ddf_train,
            group_col="country",
            name="country"
        )
        _save_lookup_table(tables["country"], output_dir / "country_stats.json")
    
    # 4. Segment-level statistics (country × dev_os)
    if "country" in ddf_train.columns and "dev_os" in ddf_train.columns:
        LOGGER.info("Computing segment-level statistics (country × dev_os)...")
        tables["segment"] = _compute_segment_stats(ddf_train)
        _save_lookup_table(tables["segment"], output_dir / "segment_stats.json")
    
    LOGGER.info(f"✅ Generated {len(tables)} lookup tables in {output_dir}")
    
    return tables


def _compute_group_stats(
    ddf: dd.DataFrame,
    group_col: str,
    name: str
) -> Dict[str, Dict[str, float]]:
    """Compute statistics for a single grouping column.
    
    Args:
        ddf: Input DataFrame
        group_col: Column to group by
        name: Name for logging
        
    Returns:
        Dict mapping group values to stats
    """
    # Group and aggregate
    agg_result = ddf.groupby(group_col).agg({
        "buyer_d7": ["mean", "count"],
        "iap_revenue_d7": "mean",
        "is_whale": "mean"
    }).compute()
    
    # Flatten multi-index columns
    agg_result.columns = ["_".join(col).strip("_") for col in agg_result.columns.values]
    agg_result = agg_result.reset_index()
    
    # Rename for clarity
    stats_dict = {}
    for _, row in agg_result.iterrows():
        key = str(row[group_col])
        stats_dict[key] = {
            "buyer_rate": float(row.get("buyer_d7_mean", 0.0)),
            "avg_rev_d7": float(row.get("iap_revenue_d7_mean", 0.0)),
            "whale_rate": float(row.get("is_whale_mean", 0.0)),
            "count": int(row.get("buyer_d7_count", 0))
        }
    
    LOGGER.info(f"  {name}: {len(stats_dict)} groups")
    
    return stats_dict


def _compute_segment_stats(ddf: dd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute statistics for (country, dev_os) segments.
    
    Args:
        ddf: Input DataFrame
        
    Returns:
        Dict mapping "country_dev_os" to stats
    """
    # Create segment key
    ddf = ddf.assign(
        segment=ddf["country"].astype(str) + "_" + ddf["dev_os"].astype(str)
    )
    
    # Group and aggregate
    agg_result = ddf.groupby("segment").agg({
        "buyer_d7": ["mean", "count"],
        "iap_revenue_d7": "mean",
        "is_whale": "mean"
    }).compute()
    
    # Flatten multi-index columns
    agg_result.columns = ["_".join(col).strip("_") for col in agg_result.columns.values]
    agg_result = agg_result.reset_index()
    
    # Convert to dict
    stats_dict = {}
    for _, row in agg_result.iterrows():
        key = str(row["segment"])
        stats_dict[key] = {
            "buyer_rate": float(row.get("buyer_d7_mean", 0.0)),
            "avg_rev_d7": float(row.get("iap_revenue_d7_mean", 0.0)),
            "whale_rate": float(row.get("is_whale_mean", 0.0)),
            "count": int(row.get("buyer_d7_count", 0))
        }
    
    LOGGER.info(f"  segment: {len(stats_dict)} groups")
    
    return stats_dict


def _save_lookup_table(table: Dict, path: Path) -> None:
    """Save lookup table as JSON.
    
    Args:
        table: Lookup table dict
        path: Output file path
    """
    with open(path, "w") as f:
        json.dump(table, f, indent=2)
    
    LOGGER.info(f"  Saved to {path}")


def load_lookup_tables(path: str | Path) -> Dict[str, Dict]:
    """Load precomputed lookup tables from disk.
    
    Args:
        path: Path to directory containing lookup table JSON files
        
    Returns:
        Dict with keys ['bundle', 'category', 'country', 'segment']
        
    Example:
        >>> from src.features.lookup_tables import load_lookup_tables
        >>> lookup_tables = load_lookup_tables("data/processed/lookup_tables")
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



