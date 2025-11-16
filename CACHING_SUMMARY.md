# Caching Summary

## ✅ Implemented Caching

### 1. **Feature Caching** (Implemented)
- **Location**: `data/processed/features_cache/`
- **What's cached**: Computed features (X_train, X_val) and targets (y_cls_train, y_cls_val, y_reg_train, y_reg_val)
- **Cache key**: `v2` - change this when feature logic changes to invalidate cache
- **Speedup**: ~5 minutes → <1 second

### 2. **Model Caching** (Implemented)
- **CatBoost Classifier**: `models/teachers/teacher_classifier_catboost.cbm`
  - Speedup: ~12 minutes → <1 second
- **LightGBM Regressor**: `models/teachers/teacher_regressor_lgb_d7.txt`
  - Speedup: ~10 seconds → <1 second

### 3. **Encoder Caching** (Already existed)
- **Location**: `data/processed/encoders/online_encoders.pkl`
- **What's cached**: Ordinal encoders for categorical features

## Performance

| Run Type | Time | Notes |
|----------|------|-------|
| First run (no cache) | ~15 minutes | Feature extraction (5 min) + CatBoost (12 min) + LightGBM (10s) |
| Second run (all cached) | <10 seconds | Just loads from disk |

## How to Clear Cache

```bash
# Clear feature cache (forces recomputation)
rm -rf data/processed/features_cache/

# Clear model cache (forces retraining)
rm models/teachers/teacher_*.cbm
rm models/teachers/teacher_*.txt

# Clear encoder cache
rm data/processed/encoders/online_encoders.pkl
```

## Progressive Feature Caching (TODO)

**Current limitation**: Feature caching is all-or-nothing. If you train on 1 day, then 2 days, it recomputes everything.

**Desired behavior**: Cache features per day, so day 1 features can be reused when training on days 1-2.

**Implementation approach**:
1. Change cache structure to: `features_cache/day_YYYY-MM-DD/train.parquet`
2. When loading data for multiple days, check which days are cached
3. Only compute features for uncached days
4. Concatenate cached + newly computed features

This would allow incremental scaling from 1→7 days without recomputing everything.

## Cache Invalidation

The cache key (`v2`) should be updated whenever:
- Feature extraction logic changes in `src/features/online.py` or `src/features/offline.py`
- Lookup table generation changes in `src/features/lookup_tables.py`
- Data preprocessing changes

Model cache is automatically invalidated by deleting the model files.
