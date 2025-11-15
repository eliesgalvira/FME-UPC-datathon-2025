"""Feature engineering modules for teacher and student models.

This package provides:
- online.py: Fast features for student models (< 100μs per prediction)
- offline.py: Rich features for teacher models (offline training, slow OK)
- lookup_tables.py: Precomputed statistics for fast inference
"""

from __future__ import annotations

__all__ = ["online", "offline", "lookup_tables"]



