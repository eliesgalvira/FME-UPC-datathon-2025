# Dask Best Practices Implementation

This document explains the Dask best practices implemented in the codebase, based on the [official Dask documentation](https://docs.dask.org/en/stable/best-practices.html).

## Summary of Best Practices Applied

### 1. ✅ Load Data with Dask
**Practice**: Never create large objects on the client and pass them to Dask. Instead, use Dask to load the data directly.

**Implementation**:
- `DataLoader._read()` uses `dd.read_parquet()` directly
- Filters are passed to the read operation for partition pruning
- Data stays in Dask format until explicitly computed

**Example**:
```python
# ❌ BAD: Loading with pandas first
df = pd.read_parquet("data.parquet")
ddf = dd.from_pandas(df, npartitions=10)

# ✅ GOOD: Let Dask read directly
ddf = dd.read_parquet("data.parquet", engine='pyarrow')
```

### 2. ✅ Avoid Calling compute() Repeatedly
**Practice**: Calling `compute()` in a loop is very inefficient. Build up the computation graph and call `compute()` once.

**Implementation**:
- `iter_batches()`: Computes 10 partitions at once using `dask.compute(*batch)`
- `encode_string_columns()`: Computes all unique values in a single call
- `materialize()`: Single `compute()` call at the end

**Example**:
```python
# ❌ BAD: Multiple compute calls
results = []
for i in range(...):
    results.append(ddf.select(...).compute())

# ✅ GOOD: Build graph, compute once
results = []
for i in range(...):
    results.append(ddf.select(...))
results = dask.compute(*results)
```

### 3. ✅ Use Partition Pruning with Filters
**Practice**: Use filters to read only the partitions you need. This is extremely fast!

**Implementation**:
- `load_train()`: Uses filters for train/val split
- `load_test()`: Uses filters for test window
- Filters on the `datetime` partition column enable instant filtering

**Example**:
```python
# ✅ GOOD: Partition pruning (instant!)
filters = [("datetime", ">=", "2025-10-01"), ("datetime", "<=", "2025-10-02")]
ddf = dd.read_parquet(path, filters=filters)
```

### 4. ✅ Monitor Partition Sizes
**Practice**: Partitions should be 100-500MB. Too large = memory issues. Too small = overhead.

**Implementation**:
- `_read()`: Automatically checks partition sizes and warns if they're too large/small
- Logs average partition size for monitoring

**Target**: 100-500 MB per partition

### 5. ✅ Dashboard for Monitoring
**Practice**: Use the Dask dashboard to understand what's happening.

**Implementation**:
- `_ensure_client()`: Logs dashboard URL prominently
- Dashboard link always available via `client.dashboard_link`

**Access**: Check logs for dashboard URL (e.g., `http://localhost:8787`)

### 6. ✅ Threads vs Processes Configuration
**Practice**: 
- **Numeric work** (NumPy, pandas): Use ~4 threads per worker
- **Text/Python objects**: Use more workers with fewer threads

**Implementation**:
- Default: 4 threads per worker (good for numeric work)
- Configurable via config file for text-heavy workloads
- Leaves CPUs for OS (doesn't use all available cores)

**Configuration**:
```yaml
dask:
  client:
    n_workers: 4           # Number of worker processes
    threads_per_worker: 4  # Threads per worker (good for numeric)
    memory_limit: "4GB"    # Memory limit per worker
```

### 7. ✅ Persist Strategically
**Practice**: Use `persist()` for intermediate results that will be used multiple times.

**Implementation**:
- `materialize()`: Option to persist before computing
- Useful when the same filtered data is used for multiple operations

**Example**:
```python
# If using filtered data multiple times
ddf_filtered = ddf[ddf['datetime'] > '2025-10-01']
ddf_filtered = ddf_filtered.persist()  # Cache in memory

# Now these are fast (data already computed)
stats = ddf_filtered.describe().compute()
counts = ddf_filtered['column'].value_counts().compute()
```

### 8. ✅ String Column Handling
**Practice**: String columns are VERY SLOW in Dask. Convert to numeric codes for ML.

**Implementation**:
- PyArrow strings kept by default (efficient for I/O)
- `encode_string_columns()`: Helper to convert strings to numeric codes
- Notebook examples show how to encode categorical columns

**Example**:
```python
# Encode string columns to numeric
ddf_encoded, mappings = loader.encode_string_columns(
    ddf, 
    columns=['dev_os', 'country', 'carrier']
)
```

## Performance Tips

### Start Small
Before using Dask:
1. **Try better algorithms** - NumPy/pandas may have faster functions
2. **Use better file formats** - Parquet with snappy compression
3. **Sample your data** - Often don't need all 20M rows
4. **Profile your code** - Understand what's slow before parallelizing

### Avoid Very Large Partitions
- Target: 100-500 MB per partition
- Too large: Memory pressure, worker crashes
- Too small: Excessive overhead, slow scheduling

### Avoid Very Large Graphs
- Don't create millions of tasks
- Use `blocksize` parameter when reading
- Consider repartitioning if graph gets too large

## Common Anti-Patterns to Avoid

### ❌ DON'T: Compute in a loop
```python
for partition in ddf.to_delayed():
    result = partition.compute()  # SLOW!
```

### ✅ DO: Batch compute calls
```python
delayed_parts = ddf.to_delayed()
for i in range(0, len(delayed_parts), 10):
    batch = delayed_parts[i:i+10]
    results = dask.compute(*batch)  # Fast!
```

### ❌ DON'T: Load large data on client
```python
df = pd.read_csv("huge.csv")  # Loads on client
ddf = dd.from_pandas(df, npartitions=10)  # Sends to workers
```

### ✅ DO: Load with Dask
```python
ddf = dd.read_csv("huge.csv")  # Workers read directly
```

### ❌ DON'T: Work with strings in Dask
```python
# String operations are VERY slow
result = ddf.groupby('string_column').mean().compute()
```

### ✅ DO: Convert strings to numeric first
```python
# Much faster with numeric codes
ddf['column_encoded'] = ddf['string_column'].map(mapping)
result = ddf.groupby('column_encoded').mean().compute()
```

## Configuration Best Practices

### Example Configuration (config.yaml)
```yaml
dask:
  client:
    enabled: true
    n_workers: null  # Auto-detect (cores / 4)
    threads_per_worker: 4  # Good for numeric work
    memory_limit: "4GB"
  
  read:
    chunksize: null  # Let Dask decide
  
  materialization:
    train:
      sample_frac: 0.1  # Start small!
      persist: true
    val:
      persist: true
    test:
      persist: false

data:
  train_path: "data/raw/train/train"
  test_path: "data/raw/test/test"
  train_start: "2025-10-01-00-00"
  train_end: "2025-10-04-23-00"
  val_start: "2025-10-05-00-00"
  val_end: "2025-10-05-23-00"
  test_start: "2025-10-08-00-00"
  test_end: "2025-10-08-23-00"
```

## Monitoring Performance

### Using the Dashboard
1. Check logs for dashboard URL
2. Open in browser (usually http://localhost:8787)
3. Monitor:
   - Task progress
   - Memory usage per worker
   - Network communication
   - Task bottlenecks

### Key Metrics to Watch
- **Memory**: Should stay below limits
- **Task duration**: Consistent is good, spiky is bad
- **Worker utilization**: Should be high during compute
- **Network**: Minimal is better

## Testing and Validation

Always test with small samples first:
```python
# Start with 10% sample
loader = DataLoader(config)
train_ddf, val_ddf = loader.load_train()
train_sample = loader.materialize(train_ddf, sample_frac=0.1)

# Verify results look good
print(train_sample.shape)
print(train_sample.dtypes)

# Scale up gradually
train_full = loader.materialize(train_ddf)  # Now load full data
```

## References

- [Dask Best Practices](https://docs.dask.org/en/stable/best-practices.html)
- [DataFrame Best Practices](https://docs.dask.org/en/stable/dataframe-best-practices.html)
- [Understanding Performance](https://docs.dask.org/en/stable/understanding-performance.html)
- [Dask Dashboard](https://docs.dask.org/en/stable/dashboard.html)

## Need Help?

If performance is still slow:
1. Check the dashboard - what's the bottleneck?
2. Profile your code - is Dask even the issue?
3. Try sampling - do you need all the data?
4. Check partition sizes - are they in the 100-500MB range?
5. Review string columns - did you convert them to numeric?

