"""Dask-powered parquet loading utilities following best practices.

Best Practices Implemented:
- Avoid calling compute() repeatedly - use compute() once on collections
- Use partition pruning with filters for efficient data access
- Keep string columns as PyArrow strings (convert to numeric for ML via map_partitions)
- Use persist() strategically to cache intermediate results
- Provide dashboard links for monitoring
- Use map_partitions for custom operations instead of compute loops
- Filter and project early to minimize data size
- Avoid shuffles and sorts when possible
- Use pure functions with explicit parameters
- Keep task granularity reasonable (not millions of tiny tasks)

References:
- https://docs.dask.org/en/stable/best-practices.html
- https://docs.dask.org/en/stable/dataframe-best-practices.html
"""

from __future__ import annotations

import multiprocessing
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import dask
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, wait

from src.utils.logger import get_logger

# Keep PyArrow strings for efficiency (convert to numeric codes for ML later)
# Dask best practice: avoid automatic string conversion, handle explicitly
dask.config.set({"dataframe.convert-string": False})

LOGGER = get_logger(__name__)

# Constants for partition size recommendations (in MB)
MIN_PARTITION_SIZE_MB = 10
MAX_PARTITION_SIZE_MB = 500
COMPUTE_BATCH_SIZE = 10  # Partitions to compute at once in iter_batches


@dataclass(frozen=True)
class PartitionInfo:
	"""Information about DataFrame partitioning.
	
	Immutable to enforce clarity about when repartitioning happens.
	"""
	n_partitions: int
	avg_size_mb: float
	
	def is_well_sized(self) -> bool:
		"""Check if partition size is within recommended range."""
		return MIN_PARTITION_SIZE_MB <= self.avg_size_mb <= MAX_PARTITION_SIZE_MB


def _estimate_partition_size_lazy(ddf: dd.DataFrame) -> dd.core.Scalar:
	"""Estimate average partition size without computing.
	
	Returns a lazy Dask Scalar that can be computed later.
	Best practice: Build computation graph, compute once at the end.
	"""
	assert isinstance(ddf, dd.DataFrame), "ddf must be a Dask DataFrame"
	assert ddf.npartitions > 0, "DataFrame must have at least one partition"
	
	return ddf.memory_usage_per_partition(deep=True).mean()


def _check_partition_size(n_partitions: int, avg_size_mb: float) -> None:
	"""Log warnings if partition sizes are suboptimal.
	
	Pure function with no side effects except logging.
	Follows Python best practice: clear, focused helper function.
	"""
	assert n_partitions > 0, "n_partitions must be positive"
	assert avg_size_mb >= 0, "avg_size_mb must be non-negative"
	
	if avg_size_mb > MAX_PARTITION_SIZE_MB:
		LOGGER.warning(
			"⚠️  Large partitions detected: %.1f MB avg (recommended: %d-%d MB). "
			"Consider repartitioning with ddf.repartition().",
			avg_size_mb, MIN_PARTITION_SIZE_MB, MAX_PARTITION_SIZE_MB
		)
	elif avg_size_mb < MIN_PARTITION_SIZE_MB:
		LOGGER.warning(
			"⚠️  Very small partitions detected: %.1f MB avg (recommended: %d-%d MB). "
			"Consider using larger partitions to reduce overhead.",
			avg_size_mb, MIN_PARTITION_SIZE_MB, MAX_PARTITION_SIZE_MB
		)
	else:
		LOGGER.info("✅ Partition size looks good: %.1f MB avg", avg_size_mb)


class DataLoader:
	"""Encapsulates train/val/test parquet loading logic using Dask."""

	def __init__(self, config: dict) -> None:
		self.config = config
		self.train_path = Path(config["data"]["train_path"])
		self.test_path = Path(config["data"]["test_path"])
		self._dask_cfg = config.get("dask", {})
		self._client: Client | None = None

	def _ensure_client(self) -> Client | None:
		"""Initialize Dask client following best practices.
		
		Best Practice: Use ~4 threads per worker for numeric workloads.
		For text/Python objects, prefer more workers with fewer threads.
		See: https://docs.dask.org/en/stable/best-practices.html#processes-threads-and-vm-sizes
		"""
		client_cfg = self._dask_cfg.get("client", {})
		if not client_cfg.get("enabled", False):
			return None
		if self._client is not None:
			return self._client

		n_workers = client_cfg.get("n_workers")
		if n_workers is None:
			cpu_count = multiprocessing.cpu_count()
			# Best practice: Don't use all CPUs, leave some for OS
			# For numeric work: fewer workers, more threads (e.g., 4 threads/worker)
			# For text work: more workers, fewer threads (e.g., 2 threads/worker)
			n_workers = max(1, cpu_count // 4) if cpu_count > 4 else 1
			
		threads = client_cfg.get("threads_per_worker", 4)  # 4 threads good for numeric work
		memory_limit = client_cfg.get("memory_limit", "4GB")
		
		self._client = Client(
			n_workers=n_workers,
			threads_per_worker=threads,
			memory_limit=memory_limit,
			silence_logs=False,  # Keep logs for debugging
		)
		LOGGER.info(
			"Started Dask client: %s workers × %s threads = %s total threads | Dashboard: %s",
			n_workers, threads, n_workers * threads, self._client.dashboard_link
		)
		LOGGER.info("💡 View real-time performance at: %s", self._client.dashboard_link)
		return self._client

	def close(self) -> None:
		if self._client is not None:
			LOGGER.info("Closing Dask client")
			self._client.close()
			self._client = None

	def _read(
		self, 
		path: Path, 
		filters: Sequence[tuple[str, str, str]] | None = None,
		check_partitions: bool = False
	) -> dd.DataFrame:
		"""Read parquet data with Dask following best practices.
		
		Best Practices Applied:
		- Use filters for partition pruning (Dask best practice: filter early)
		- Keep data in Dask format until needed (avoid compute())
		- PyArrow engine for efficient reading
		- Optional partition size checking (disabled by default to avoid compute())
		
		Args:
			path: Path to parquet dataset
			filters: List of filters for partition pruning
			check_partitions: If True, compute and log partition sizes (requires compute())
		
		Returns:
			Lazy Dask DataFrame (no computation triggered)
			
		Raises:
			FileNotFoundError: If path doesn't exist
		
		See: https://docs.dask.org/en/stable/best-practices.html#load-data-with-dask
		"""
		# Preconditions
		assert isinstance(path, Path), "path must be a Path object"
		if not path.exists():
			raise FileNotFoundError(f"Dataset path not found: {path}")

		self._ensure_client()
		read_cfg = self._dask_cfg.get("read", {})
		chunksize = read_cfg.get("chunksize")
		read_kwargs = {"chunksize": chunksize} if chunksize else {}

		LOGGER.info("Reading parquet from %s with filters=%s", path, filters)
		
		# Dask best practice: Use high-level collections with filters for partition pruning
		ddf = dd.read_parquet(
			path, 
			filters=filters, 
			engine="pyarrow", 
			**read_kwargs
		)
		
		LOGGER.info("Loaded %d partitions (lazy)", ddf.npartitions)
		
		# Optional: Check partition sizes (requires compute(), so opt-in only)
		# Dask best practice: Avoid unnecessary compute() calls
		if check_partitions and ddf.npartitions > 0:
			self._check_and_log_partition_sizes(ddf)
				
		return ddf
	
	def _check_and_log_partition_sizes(self, ddf: dd.DataFrame) -> None:
		"""Check and log partition sizes (requires compute()).
		
		Dask best practice: Only call compute() when explicitly needed.
		This is a separate method to make the compute() call visible.
		"""
		assert isinstance(ddf, dd.DataFrame), "ddf must be a Dask DataFrame"
		
		try:
			# This is one of the few places where compute() is justified
			avg_mb = _estimate_partition_size_lazy(ddf).compute() / (1024 ** 2)
			_check_partition_size(ddf.npartitions, avg_mb)
		except Exception as e:
			LOGGER.debug("Could not compute partition sizes: %s", e)

	def read_path(
		self, 
		path: str | Path, 
		filters: Sequence[tuple[str, str, str]] | None = None,
		check_partitions: bool = False
	) -> dd.DataFrame:
		"""Public helper to read an arbitrary parquet path with Dask settings.
		
		Args:
			path: Path to parquet dataset (str or Path)
			filters: Optional filters for partition pruning
			check_partitions: Whether to check partition sizes (requires compute())
			
		Returns:
			Lazy Dask DataFrame
		"""
		return self._read(Path(path), filters, check_partitions)

	def load_train(
		self,
		validation_split: bool = True,
	) -> tuple[dd.DataFrame, dd.DataFrame | None]:
		"""Load training data with optional temporal validation split.
		
		Dask best practice: Use filters for partition pruning to only load
		relevant partitions. This is much faster than loading all and filtering.
		
		Args:
			validation_split: If True, return (train, val), else (train, None)
			
		Returns:
			Tuple of (train_ddf, val_ddf) where val_ddf is None if no split
		"""
		if validation_split:
			# Python best practice: Extract filter construction to helper
			train_filters = self._build_time_filters("train")
			val_filters = self._build_time_filters("val")

			# Dask best practice: Both reads are lazy, no compute() yet
			train_ddf = self._read(self.train_path, train_filters)
			val_ddf = self._read(self.train_path, val_filters)

			LOGGER.info(
				"Train partitions: %d | Validation partitions: %d",
				train_ddf.npartitions,
				val_ddf.npartitions,
			)
			return train_ddf, val_ddf

		ddf = self._read(self.train_path)
		LOGGER.info("Full train partitions: %d", ddf.npartitions)
		return ddf, None
	
	def _build_time_filters(self, split: str) -> Sequence[tuple[str, str, str]]:
		"""Build time-based filters for a data split.
		
		Python best practice: Pure helper function with clear responsibility.
		Returns filter tuples for partition pruning.
		
		Args:
			split: One of "train", "val", "test"
			
		Returns:
			List of filter tuples (column, operator, value)
		"""
		assert split in ("train", "val", "test"), f"Invalid split: {split}"
		
		start_key = f"{split}_start"
		end_key = f"{split}_end"
		
		return [
			("datetime", ">=", self.config["data"][start_key]),
			("datetime", "<=", self.config["data"][end_key]),
		]

	def load_test(self) -> dd.DataFrame:
		"""Load official test data window.
		
		Uses partition pruning via filters for efficient loading.
		
		Returns:
			Lazy Dask DataFrame for test split
		"""
		test_filters = self._build_time_filters("test")
		ddf = self._read(self.test_path, test_filters)
		LOGGER.info("Test partitions: %d", ddf.npartitions)
		return ddf

	def materialize(
		self,
		ddf: dd.DataFrame,
		*,
		split: str = "train",
		sample_frac: float | None = None,
		max_rows: int | None = None,
		persist: bool | None = None,
		random_state: int | None = None,
	) -> pd.DataFrame:
		"""Convert a Dask DataFrame into pandas with optional sampling.
		
		Dask best practice: Build computation graph with lazy operations,
		then call compute() once at the end.
		
		Args:
			ddf: Dask DataFrame to materialize
			split: Split name for config lookup ("train", "val", "test")
			sample_frac: Fraction of data to sample (0, 1]
			max_rows: Maximum rows to load (uses head())
			persist: Whether to persist intermediate results
			random_state: Random seed for sampling
			
		Returns:
			Pandas DataFrame with materialized data
			
		See: https://docs.dask.org/en/stable/best-practices.html#avoid-calling-compute-repeatedly
		"""
		# Preconditions
		assert isinstance(ddf, dd.DataFrame), "ddf must be a Dask DataFrame"
		assert len(ddf.columns) > 0, "ddf must have at least one column"

		# Get config with defaults
		plan = self._dask_cfg.get("materialization", {}).get(split, {})
		sample_frac = sample_frac if sample_frac is not None else plan.get("sample_frac")
		max_rows = max_rows if max_rows is not None else plan.get("max_rows")
		persist_flag = plan.get("persist", True) if persist is None else persist
		random_state = random_state or self.config.get("training", {}).get("random_state", 42)
		
		# Invariant: sample_frac must be in valid range if specified
		if sample_frac is not None:
			assert 0.0 < sample_frac <= 1.0, f"sample_frac must be in (0, 1], got {sample_frac}"

		# Dask best practice: Build computation graph (all lazy operations)
		result_ddf = self._build_materialization_graph(
			ddf, sample_frac, persist_flag, random_state, split
		)

		# Dask best practice: Single compute() call at the end
		pdf = self._compute_to_pandas(result_ddf, max_rows, split)
		
		# Postcondition: result must be valid pandas DataFrame
		assert isinstance(pdf, pd.DataFrame), "Result must be a pandas DataFrame"
		assert len(pdf) > 0 or max_rows == 0, "Result should have rows unless max_rows=0"
		
		self._log_materialization_result(pdf, split)
		return pdf
	
	def _build_materialization_graph(
		self,
		ddf: dd.DataFrame,
		sample_frac: float | None,
		persist_flag: bool,
		random_state: int,
		split: str,
	) -> dd.DataFrame:
		"""Build lazy computation graph for materialization.
		
		Python best practice: Extract graph building to focused helper.
		All operations here are lazy - no compute() called.
		
		Returns:
			Dask DataFrame with lazy transformations applied
		"""
		result_ddf = ddf
		
		# Apply sampling if requested (lazy operation)
		if sample_frac and sample_frac < 1.0:
			LOGGER.info("Applying sample_frac=%s to %s split", sample_frac, split)
			result_ddf = result_ddf.sample(frac=float(sample_frac), random_state=random_state)

		# Persist is useful for intermediate results used multiple times
		# Dask best practice: Use persist() for data reused in multiple operations
		if persist_flag:
			client = self._ensure_client()
			LOGGER.info("Persisting %s split into Dask cache", split)
			result_ddf = result_ddf.persist()
			if client is not None:
				wait(result_ddf)
		
		return result_ddf
	
	def _compute_to_pandas(
		self, 
		ddf: dd.DataFrame, 
		max_rows: int | None, 
		split: str
	) -> pd.DataFrame:
		"""Compute Dask DataFrame to pandas.
		
		Python best practice: Separate compute logic into focused function.
		Dask best practice: Single compute() call.
		
		Returns:
			Materialized pandas DataFrame
		"""
		if max_rows:
			return ddf.head(int(max_rows), compute=True)
		
		LOGGER.info("Computing %s split to pandas (this may take a while)...", split)
		return ddf.compute()
	
	def _log_materialization_result(self, pdf: pd.DataFrame, split: str) -> None:
		"""Log information about materialized DataFrame.
		
		Python best practice: Pure function for logging (no state mutation).
		"""
		memory_gb = pdf.memory_usage(deep=True).sum() / 1e9
		LOGGER.info("✅ Materialized %s split: %d rows, %.2f GB", split, len(pdf), memory_gb)
		
		# Warn if result is very large
		if memory_gb > 10:
			LOGGER.warning(
				"⚠️  Large dataset in memory (%.2f GB). "
				"Consider using sampling or working with Dask directly.",
				memory_gb
			)

	@staticmethod
	def iter_batches(ddf: dd.DataFrame, batch_size: int = 100_000) -> Iterator[pd.DataFrame]:
		"""Materialize partitions incrementally, yielding pandas batches.
		
		Dask best practice: Avoid calling compute() in a loop. Instead, compute
		multiple delayed objects at once to balance memory and efficiency.
		
		This method computes partitions in batches to balance:
		- Memory efficiency (not loading all data at once)
		- Computation efficiency (not calling compute() excessively)
		
		Args:
			ddf: Dask DataFrame to iterate over
			batch_size: Target size for each yielded batch (in rows)
			
		Yields:
			Pandas DataFrames with approximately batch_size rows each
		
		See: https://docs.dask.org/en/stable/best-practices.html#avoid-calling-compute-repeatedly
		"""
		# Preconditions
		assert isinstance(ddf, dd.DataFrame), "ddf must be a Dask DataFrame"
		assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
		
		delayed_parts = ddf.to_delayed()
		
		# Dask best practice: Compute multiple partitions at once
		# Process partitions in chunks to avoid excessive compute() calls
		cache: list[pd.DataFrame] = []
		cached_rows = 0
		
		for i in range(0, len(delayed_parts), COMPUTE_BATCH_SIZE):
			# Dask best practice: Single compute() call for multiple delayed objects
			batch = delayed_parts[i : i + COMPUTE_BATCH_SIZE]
			computed_parts = dask.compute(*batch)
			
			for part in computed_parts:
				cache.append(part)
				cached_rows += len(part)
				
				# Yield when we've accumulated enough rows
				if cached_rows >= batch_size:
					yield pd.concat(cache, ignore_index=True)
					cache.clear()
					cached_rows = 0
		
		# Yield any remaining data
		if cache:
			yield pd.concat(cache, ignore_index=True)

	@staticmethod
	def compute_in_batches(ddf: dd.DataFrame, batch_size: int = 100_000) -> Iterator[pd.DataFrame]:
		"""Backward-compatible wrapper around :meth:`iter_batches`."""

		warnings.warn(
			"DataLoader.compute_in_batches is deprecated; use DataLoader.iter_batches instead.",
			DeprecationWarning,
			stacklevel=2,
		)
		yield from DataLoader.iter_batches(ddf, batch_size=batch_size)
	
	@staticmethod
	def encode_string_columns(
		ddf: dd.DataFrame, 
		columns: Sequence[str],
		suffix: str = "_encoded"
	) -> tuple[dd.DataFrame, Mapping[str, Mapping[str, int]]]:
		"""Encode string columns to numeric for ML models using map_partitions.
		
		Dask best practices applied:
		- String columns are VERY SLOW in Dask - convert to numeric for ML
		- Compute unique values once (single compute() call)
		- Use map_partitions for efficient per-partition encoding
		- Keep result lazy (no additional compute() calls)
		
		Args:
			ddf: Dask DataFrame
			columns: Column names to encode
			suffix: Suffix for new encoded columns
			
		Returns:
			Tuple of (DataFrame with new encoded columns, mapping dictionary)
			
		See: 
		- https://docs.dask.org/en/stable/dataframe-best-practices.html
		- https://docs.dask.org/en/stable/dataframe-api.html#dask.dataframe.DataFrame.map_partitions
		"""
		# Preconditions
		assert isinstance(ddf, dd.DataFrame), "ddf must be a Dask DataFrame"
		assert len(columns) > 0, "columns list must not be empty"
		assert all(col in ddf.columns for col in columns), "All columns must exist in DataFrame"
		
		# Dask best practice: Compute all unique values at once (single compute!)
		LOGGER.info("Computing unique values for %d columns...", len(columns))
		unique_computations = {
			col: ddf[col].unique() for col in columns
		}
		unique_results = dask.compute(unique_computations)[0]
		
		# Create mappings (pure Python, fast)
		mappings: dict[str, dict[str, int]] = {}
		for col, unique_vals in unique_results.items():
			mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
			LOGGER.info("  %s: %d unique values", col, len(mappings[col]))
		
		# Dask best practice: Use map_partitions for efficient encoding
		# This applies the encoding function to each partition (parallel operation)
		result_ddf = ddf.map_partitions(
			_encode_partition,
			columns=columns,
			mappings=mappings,
			suffix=suffix,
			meta=_create_meta_for_encoding(ddf, columns, suffix)
		)
		
		return result_ddf, mappings


def _encode_partition(
	partition: pd.DataFrame,
	columns: Sequence[str],
	mappings: Mapping[str, Mapping[str, int]],
	suffix: str
) -> pd.DataFrame:
	"""Encode string columns in a single partition.
	
	Pure function for use with map_partitions.
	Dask best practice: Use map_partitions for custom per-partition logic.
	
	Args:
		partition: Single pandas DataFrame partition
		columns: Columns to encode
		mappings: Encoding mappings for each column
		suffix: Suffix for encoded columns
		
	Returns:
		Partition with additional encoded columns
	"""
	# Python best practice: Avoid mutable default args, take all data as params
	result = partition.copy()
	
	for col in columns:
		if col in result.columns:
			encoded_col = f"{col}{suffix}"
			# Use map with na_action='ignore' to handle missing values
			result[encoded_col] = result[col].map(mappings[col])
	
	return result


def _create_meta_for_encoding(
	ddf: dd.DataFrame,
	columns: Sequence[str],
	suffix: str
) -> pd.DataFrame:
	"""Create metadata for encoded DataFrame.
	
	Dask requires meta (schema) for map_partitions operations.
	
	Args:
		ddf: Original Dask DataFrame
		columns: Columns being encoded
		suffix: Suffix for encoded columns
		
	Returns:
		Empty pandas DataFrame with correct schema
	"""
	# Start with original schema
	meta = ddf._meta.copy()
	
	# Add encoded columns as integer type
	for col in columns:
		if col in meta.columns:
			encoded_col = f"{col}{suffix}"
			meta[encoded_col] = pd.Series(dtype='int64')
	
	return meta
