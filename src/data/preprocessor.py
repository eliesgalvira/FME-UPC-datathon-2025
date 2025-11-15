"""Utilities to flatten nested Smadex features."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


class NestedFeatureParser:
    """Transforms dict/list/ts columns into numeric aggregates."""

    @staticmethod
    def _dict_values(value: object) -> List[float]:
        if isinstance(value, dict) and value:
            return list(value.values())
        return []

    @staticmethod
    def parse_dict_feature(
        df: pd.DataFrame,
        column: str,
        aggs: Iterable[str] = ("mean", "std", "max", "min"),
    ) -> pd.DataFrame:
        if column not in df.columns:
            return df

        LOGGER.info("Parsing dict column %s", column)
        values = df[column].apply(NestedFeatureParser._dict_values)

        for agg in aggs:
            if agg == "mean":
                df[f"{column}_mean"] = values.apply(
                    lambda x: float(np.mean(x)) if x else 0.0
                )
            elif agg == "std":
                df[f"{column}_std"] = values.apply(
                    lambda x: float(np.std(x)) if len(x) > 1 else 0.0
                )
            elif agg == "max":
                df[f"{column}_max"] = values.apply(lambda x: float(np.max(x)) if x else 0.0)
            elif agg == "min":
                df[f"{column}_min"] = values.apply(lambda x: float(np.min(x)) if x else 0.0)

        df[f"{column}_count"] = values.apply(len)
        return df.drop(columns=[column])

    @staticmethod
    def parse_list_feature(df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column not in df.columns:
            return df

        LOGGER.info("Parsing list column %s", column)
        df[f"{column}_count"] = df[column].apply(
            lambda x: len(x) if isinstance(x, (list, tuple)) else 0
        )
        df[f"{column}_unique"] = df[column].apply(
            lambda x: len(set(x)) if isinstance(x, (list, tuple)) else 0
        )
        return df.drop(columns=[column])

    @staticmethod
    def parse_hist_feature(df: pd.DataFrame, column: str, top_k: int = 5) -> pd.DataFrame:
        if column not in df.columns:
            return df

        LOGGER.info("Parsing histogram column %s", column)

        def to_features(hist: object) -> Dict[str, float]:
            if not isinstance(hist, dict) or not hist:
                feats = {f"top{i}_freq": 0.0 for i in range(1, top_k + 1)}
                feats["entropy"] = 0.0
                return feats

            sorted_items = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)
            feats = {}
            for i in range(top_k):
                feats[f"top{i+1}_freq"] = float(sorted_items[i][1]) if i < len(sorted_items) else 0.0

            total = float(sum(hist.values()))
            if total == 0:
                feats["entropy"] = 0.0
            else:
                probs = np.array(list(hist.values())) / total
                feats["entropy"] = float(-(probs * np.log(probs + 1e-10)).sum())
            return feats

        parsed = df[column].apply(to_features).apply(pd.Series)
        parsed.columns = [f"{column}_{c}" for c in parsed.columns]
        return pd.concat([df.drop(columns=[column]), parsed], axis=1)

    @staticmethod
    def parse_timestamp_feature(
        df: pd.DataFrame,
        column: str,
        reference_date: str = "2025-10-08",
    ) -> pd.DataFrame:
        if column not in df.columns:
            return df

        ref = pd.to_datetime(reference_date)
        parsed = pd.to_datetime(df[column], errors="coerce")
        days_ago = (ref - parsed).dt.days.fillna(9999)
        df[f"{column}_days_ago"] = days_ago
        df[f"{column}_recency_weight"] = np.exp(-days_ago / 30.0)
        return df.drop(columns=[column])

    def process_all(self, df: pd.DataFrame) -> pd.DataFrame:
        dict_cols = [
            "iap_revenue_usd_bundle",
            "iap_revenue_usd_category",
            "num_buys_bundle",
            "num_buys_category",
            "cpm",
            "ctr",
            "cpm_pct_rk",
            "ctr_pct_rk",
            "rev_by_adv",
            "whale_users_bundle_num_buys_prank",
            "whale_users_bundle_revenue_prank",
        ]
        for col in dict_cols:
            df = self.parse_dict_feature(df, col)

        list_cols = ["bundles_ins", "new_bundles"]
        for col in list_cols:
            df = self.parse_list_feature(df, col)

        hist_cols = [
            "country_hist",
            "region_hist",
            "city_hist",
            "dev_osv_hist",
            "dev_language_hist",
        ]
        for col in hist_cols:
            df = self.parse_hist_feature(df, col)

        ts_cols = [
            "last_buy",
            "last_buy_ts_bundle",
            "first_request_ts",
            "last_ins",
            "last_install_ts_bundle",
        ]
        for col in ts_cols:
            df = self.parse_timestamp_feature(df, col)

        return df
"""
Preprocessing for nested data structures (dicts, lists, tuples)
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class NestedFeatureParser:
    """Parse complex nested features in Smadex dataset"""
    
    @staticmethod
    def parse_dict_features(
        df: pd.DataFrame, 
        column: str, 
        aggregations: List[str] | None = None
    ) -> pd.DataFrame:
        """Parse dictionary/map columns (e.g., iap_revenue_usd_bundle).
        
        Note: Function is >40 lines but cohesive - extracts dictionary values
        and computes multiple aggregation statistics in a single pass over the data.
        """
        # Preconditions
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        
        if column not in df.columns:
            return df
        
        # Fix mutable default argument (cursorrules compliance)
        if aggregations is None:
            aggregations = ['mean', 'std', 'max', 'min']
        
        # Invariant: aggregations must be non-empty
        assert len(aggregations) > 0, "aggregations list must not be empty"
        
        logger.info(f"Parsing dict feature: {column}")
        
        # Extract values from dictionaries
        values = df[column].apply(
            lambda x: list(x.values()) if isinstance(x, dict) and x else []
        )
        
        # Compute aggregations
        for agg in aggregations:
            if agg == 'mean':
                df[f'{column}_mean'] = values.apply(
                    lambda x: np.mean(x) if x else 0.0
                )
            elif agg == 'std':
                df[f'{column}_std'] = values.apply(
                    lambda x: np.std(x) if len(x) > 1 else 0.0
                )
            elif agg == 'max':
                df[f'{column}_max'] = values.apply(
                    lambda x: np.max(x) if x else 0.0
                )
            elif agg == 'min':
                df[f'{column}_min'] = values.apply(
                    lambda x: np.min(x) if x else 0.0
                )
        
        # Count non-zero entries
        df[f'{column}_count'] = values.apply(len)
        
        # Drop original column to save memory
        df = df.drop(columns=[column])
        
        return df
    
    @staticmethod
    def parse_list_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Parse list columns (e.g., bundles_ins)"""
        if column not in df.columns:
            return df
        
        logger.info(f"Parsing list feature: {column}")
        
        df[f'{column}_count'] = df[column].apply(
            lambda x: len(x) if isinstance(x, (list, tuple)) else 0
        )
        
        df[f'{column}_unique'] = df[column].apply(
            lambda x: len(set(x)) if isinstance(x, (list, tuple)) else 0
        )
        
        df = df.drop(columns=[column])
        return df
    
    @staticmethod
    def parse_histogram_features(
        df: pd.DataFrame,
        column: str,
        top_k: int = 5
    ) -> pd.DataFrame:
        """Parse histogram columns (e.g., country_hist).
        
        Note: Function is >40 lines but cohesive - extracts top-K frequencies
        and entropy from histogram dictionaries in a single logical flow.
        """
        # Precondition: top_k must be positive
        assert top_k > 0, f"top_k must be positive, got {top_k}"
        
        if column not in df.columns:
            return df
        
        logger.info(f"Parsing histogram feature: {column}")
        
        def extract_top_k_and_entropy(hist: Dict[str, int]) -> Dict[str, float]:
            if not isinstance(hist, dict) or not hist:
                return {f'top{i+1}_freq': 0.0 for i in range(top_k)} | {'entropy': 0.0}
            
            # Sort by frequency
            sorted_items = sorted(hist.items(), key=lambda x: x[1], reverse=True)
            
            # Top-K frequencies
            features = {}
            for i in range(top_k):
                if i < len(sorted_items):
                    features[f'top{i+1}_freq'] = sorted_items[i][1]
                else:
                    features[f'top{i+1}_freq'] = 0.0
            
            # Entropy (measure of diversity)
            total = sum(hist.values())
            if total > 0:
                probs = np.array(list(hist.values())) / total
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                features['entropy'] = entropy
            else:
                features['entropy'] = 0.0
            
            return features
        
        # Apply extraction
        hist_features = df[column].apply(extract_top_k_and_entropy)
        hist_df = pd.DataFrame(hist_features.tolist())
        
        # Add prefix
        hist_df.columns = [f'{column}_{col}' for col in hist_df.columns]
        
        # Concat and drop original
        df = pd.concat([df, hist_df], axis=1)
        df = df.drop(columns=[column])
        
        return df
    
    def process_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all nested features in dataset"""
        
        # Dictionary features (purchase history)
        dict_cols = [
            'iap_revenue_usd_bundle',
            'iap_revenue_usd_category',
            'num_buys_bundle',
            'num_buys_category',
            'cpm',
            'ctr',
            'cpm_pct_rk',
            'ctr_pct_rk',
            'rev_by_adv',
            'whale_users_bundle_num_buys_prank',
            'whale_users_bundle_revenue_prank'
        ]
        
        for col in dict_cols:
            df = self.parse_dict_features(df, col)
        
        # List features (install history)
        list_cols = ['bundles_ins', 'new_bundles']
        
        for col in list_cols:
            df = self.parse_list_features(df, col)
        
        # Histogram features
        hist_cols = [
            'country_hist',
            'region_hist',
            'city_hist',
            'dev_osv_hist',
            'dev_language_hist'
        ]
        
        for col in hist_cols:
            df = self.parse_histogram_features(df, col)
        
        return df
