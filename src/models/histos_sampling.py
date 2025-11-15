"""HistOS-like sampling for revenue prediction.

Problem: Revenue distribution is heavily skewed
- Many zeros (non-buyers)
- Few whales (high spenders)
- Standard training underrepresents whales → poor predictions for high-value users

Solution: Oversample high-revenue bins to give model more signal from whales.

Reference: HistOS (Histogram-based Oversampling)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def histos_sample(
    df: pd.DataFrame,
    revenue_col: str = "iap_revenue_d7",
    bins: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """Apply HistOS-like sampling for revenue prediction.
    
    Oversamples high-revenue examples to balance the distribution.
    Critical for teacher regressor to learn whale behavior.
    
    Args:
        df: Input DataFrame
        revenue_col: Revenue column name (original scale, not log)
        bins: Revenue bin edges (original scale). Defaults to [0, 1, 3, 6, 10, inf]
        weights: Sampling weight for each bin. Defaults to [0.3, 1.0, 2.0, 3.0, 10.0]
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame with overrepresentation of whales
        
    Example:
        >>> from src.models.histos_sampling import histos_sample
        >>> 
        >>> # Load data
        >>> df_train = ...
        >>> 
        >>> # Apply HistOS sampling
        >>> df_sampled = histos_sample(
        ...     df_train,
        ...     revenue_col="iap_revenue_d7",
        ...     bins=[0, 1, 3, 6, 10, np.inf],
        ...     weights=[0.3, 1.0, 2.0, 3.0, 10.0]
        ... )
        >>> 
        >>> # Now train model on df_sampled
        >>> X_train = df_sampled.drop(columns=["iap_revenue_d7"])
        >>> y_train = np.log1p(df_sampled["iap_revenue_d7"])
    
    Notes:
        - Bins are on original revenue scale, not log scale
        - Weight < 1.0 means undersample (e.g., 0.3 = keep 30%)
        - Weight > 1.0 means oversample with replacement
        - This creates a more balanced training set for the regressor
    """
    # Default bins and weights (tuned from EDA insights)
    if bins is None:
        bins = [0, 1, 3, 6, 10, np.inf]
    
    if weights is None:
        # Low revenue (0-1): undersample to 30%
        # Mid revenue (1-3): keep 100%
        # High revenue (3-6): oversample 2x
        # Very high (6-10): oversample 3x
        # Whales (>10): oversample 10x
        weights = [0.3, 1.0, 2.0, 3.0, 10.0]
    
    if len(weights) != len(bins) - 1:
        raise ValueError(f"weights length ({len(weights)}) must equal bins length - 1 ({len(bins) - 1})")
    
    LOGGER.info("Applying HistOS sampling...")
    LOGGER.info(f"  Revenue bins: {bins}")
    LOGGER.info(f"  Sampling weights: {weights}")
    LOGGER.info(f"  Original size: {len(df):,} rows")
    
    df = df.copy()
    
    # Assign bins
    df["_rev_bin"] = pd.cut(
        df[revenue_col],
        bins=bins,
        labels=range(len(bins) - 1),
        include_lowest=True,
        duplicates="drop"
    )
    
    # Sample each bin according to weights
    sampled_dfs = []
    bin_counts = []
    
    for bin_idx, weight in enumerate(weights):
        bin_df = df[df["_rev_bin"] == bin_idx]
        n_original = len(bin_df)
        
        if n_original == 0:
            LOGGER.warning(f"  Bin {bin_idx} [{bins[bin_idx]:.1f}, {bins[bin_idx + 1]:.1f}): empty")
            continue
        
        # Calculate number of samples
        n_samples = int(n_original * weight)
        
        if n_samples == 0:
            LOGGER.warning(f"  Bin {bin_idx} [{bins[bin_idx]:.1f}, {bins[bin_idx + 1]:.1f}): weight too low, skipping")
            continue
        
        # Sample (with replacement if weight > 1)
        replace = (weight > 1.0)
        sampled = bin_df.sample(
            n=n_samples,
            replace=replace,
            random_state=random_state
        )
        
        sampled_dfs.append(sampled)
        bin_counts.append((bin_idx, n_original, n_samples, weight))
        
        LOGGER.info(
            f"  Bin {bin_idx} [{bins[bin_idx]:.1f}, {bins[bin_idx + 1]:.1f}): "
            f"{n_original:,} → {n_samples:,} (weight={weight:.1f})"
        )
    
    # Combine all sampled bins
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    # Drop temporary column
    result = result.drop(columns=["_rev_bin"])
    
    # Shuffle
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    LOGGER.info(f"  Final size: {len(result):,} rows ({len(result) / len(df):.2f}x)")
    
    # Log distribution comparison
    LOGGER.info("\nRevenue distribution comparison:")
    LOGGER.info(f"  Original mean: ${df[revenue_col].mean():.2f}")
    LOGGER.info(f"  Sampled mean: ${result[revenue_col].mean():.2f}")
    LOGGER.info(f"  Original median: ${df[revenue_col].median():.2f}")
    LOGGER.info(f"  Sampled median: ${result[revenue_col].median():.2f}")
    LOGGER.info(f"  Original whales (>${bins[-2]:.0f}): {(df[revenue_col] > bins[-2]).sum():,} ({(df[revenue_col] > bins[-2]).mean():.2%})")
    LOGGER.info(f"  Sampled whales (>${bins[-2]:.0f}): {(result[revenue_col] > bins[-2]).sum():,} ({(result[revenue_col] > bins[-2]).mean():.2%})")
    
    return result


def stratified_sample_by_revenue(
    df: pd.DataFrame,
    revenue_col: str = "iap_revenue_d7",
    n_strata: int = 5,
    sample_frac: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """Stratified sampling based on revenue quantiles.
    
    Alternative to HistOS: ensures each revenue quantile is equally represented.
    Useful for validation sets.
    
    Args:
        df: Input DataFrame
        revenue_col: Revenue column name
        n_strata: Number of strata (quantiles)
        sample_frac: Fraction to sample from each stratum
        random_state: Random seed
        
    Returns:
        Stratified sample DataFrame
        
    Example:
        >>> # Create a small validation set with balanced revenue distribution
        >>> df_val_small = stratified_sample_by_revenue(
        ...     df_val,
        ...     n_strata=5,
        ...     sample_frac=0.2
        ... )
    """
    LOGGER.info(f"Creating stratified sample with {n_strata} strata...")
    
    df = df.copy()
    
    # Create strata based on quantiles
    df["_stratum"] = pd.qcut(
        df[revenue_col],
        q=n_strata,
        labels=False,
        duplicates="drop"
    )
    
    # Sample from each stratum
    sampled_dfs = []
    for stratum in range(n_strata):
        stratum_df = df[df["_stratum"] == stratum]
        n_samples = max(1, int(len(stratum_df) * sample_frac))
        sampled = stratum_df.sample(n=n_samples, random_state=random_state)
        sampled_dfs.append(sampled)
        LOGGER.info(f"  Stratum {stratum}: {len(stratum_df):,} → {len(sampled):,}")
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    result = result.drop(columns=["_stratum"])
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    LOGGER.info(f"✅ Stratified sample created: {len(result):,} rows")
    
    return result



