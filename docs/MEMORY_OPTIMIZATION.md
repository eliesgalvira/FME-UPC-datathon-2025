# Memory Optimization Guide

## Problem

The original training pipeline was loading too much data at once, causing RAM issues even on systems with 32GB of memory.

## Solution

Added configurable training days to allow gradual scaling from 1 day to 7 days of data.

### New Command-Line Options

```bash
# Option 1: Use --days flag (1-7 days)
uv run python scripts/train_teachers.py --days 1  # ~3M rows, ~14GB RAM
uv run python scripts/train_teachers.py --days 2  # ~6M rows, ~20GB RAM
uv run python scripts/train_teachers.py --days 3  # ~9M rows, ~26GB RAM
uv run python scripts/train_teachers.py --days 5  # ~15M rows, ~32GB+ RAM

# Option 2: Use mode flags (defaults to 1 day for subset, 5 days for full)
uv run python scripts/train_teachers.py --subset  # 1 day
uv run python scripts/train_teachers.py --full    # 5 days
```

### Memory Usage by Days

| Days | Rows (approx) | RAM Usage (approx) | Recommended System RAM |
|------|---------------|-------------------|------------------------|
| 1    | ~3M           | ~14GB             | 16GB+                  |
| 2    | ~6M           | ~20GB             | 24GB+                  |
| 3    | ~9M           | ~26GB             | 32GB+                  |
| 4    | ~12M          | ~32GB             | 40GB+                  |
| 5    | ~15M          | ~38GB             | 48GB+                  |
| 6    | ~18M          | ~44GB             | 56GB+                  |
| 7    | ~21M          | ~50GB             | 64GB+                  |

**Note**: These are rough estimates. Actual memory usage depends on:
- Feature complexity
- Number of unique categorical values
- Dask configuration
- System overhead

### Recommendations

1. **Start small**: Always test with `--days 1` first
2. **Monitor memory**: Use `free -h` or `htop` to watch RAM usage
3. **Scale gradually**: Increase days only if you have enough RAM
4. **Use swap carefully**: Swap can prevent crashes but will be very slow

### Code Changes

#### 1. Added `--days` Argument

```python
parser.add_argument(
    "--days",
    type=int,
    default=None,
    help="Number of training days (1-7). Overrides mode. Default: 1 for subset, 5 for full"
)
```

#### 2. Dynamic Date Calculation

```python
# Determine number of training days
if args.days is not None:
    num_days = args.days
    assert 1 <= num_days <= 7, "Number of days must be between 1 and 7"
else:
    # Default based on mode
    num_days = 1 if args.mode == "subset" else 5

# Calculate end date based on number of days
train_start = "2025-10-01-00-00"
if num_days == 1:
    train_end = "2025-10-01-23-00"
elif num_days == 2:
    train_end = "2025-10-02-23-00"
# ... etc
```

### Additional Memory Optimizations

#### 1. Fixed Encoder Sampling

**Problem**: `result.head(n=100000, npartitions=3, compute=True)` was hanging due to Dask metadata inference issues.

**Solution**: Use `get_partition(0).compute()` to sample only the first partition:

```python
# Before (problematic)
sample_df = result.head(n=100000, npartitions=n_sample_partitions, compute=True)

# After (fixed)
sample_df = result.get_partition(0).compute()
```

This is faster and more memory-efficient.

#### 2. Dask Configuration

The training script disables the Dask distributed client by default to avoid memory overhead:

```python
config = {
    "dask": {
        "client": {
            "enabled": False,  # Disabled to avoid memory issues
            "n_workers": 4,
            "threads_per_worker": 4
        }
    }
}
```

### Troubleshooting

#### Process Hangs

If the training hangs:
1. Check memory usage: `free -h`
2. If swap is being used heavily, you're out of RAM
3. Kill the process: `pkill -f train_teachers.py`
4. Reduce number of days

#### Out of Memory (OOM)

If you get OOM errors:
1. Reduce `--days` to 1
2. Close other applications
3. Consider adding swap space (slow but prevents crashes)
4. Upgrade RAM if training on larger datasets regularly

#### Slow Performance

If training is very slow:
1. Check if swap is being used: `free -h`
2. Reduce number of days
3. Enable Dask distributed client for better parallelization (but uses more memory)

### Future Improvements

Potential optimizations for even lower memory usage:

1. **Streaming computation**: Process data in smaller chunks
2. **Feature selection**: Drop low-importance features early
3. **Data types**: Use smaller dtypes (int8, int16 instead of int64)
4. **Incremental learning**: Train on batches instead of all data at once
5. **External memory**: Use Dask's disk-based storage for intermediate results

### Summary

- ✅ Training now supports 1-7 days of data
- ✅ Default is 1 day for subset mode (low memory)
- ✅ Memory usage scales linearly with number of days
- ✅ Fixed encoder sampling hang issue
- ✅ Documented memory requirements

**Recommendation**: Start with `--days 1` and scale up only if you have sufficient RAM.



