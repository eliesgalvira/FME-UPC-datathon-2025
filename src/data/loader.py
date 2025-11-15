"""Dask-powered parquet loading utilities following best practices.

Best Practices Implemented:
- Avoid calling compute() repeatedly - use compute() once on collections
- Use partition pruning with filters for efficient data access
- Keep string columns as PyArrow strings (efficient, but convert to numeric for ML)
- Use persist() strategically to cache intermediate results
- Provide dashboard links for monitoring
- Use map_partitions for custom operations instead of compute loops
- Warn about large partitions and memory usage

References:
- https://docs.dask.org/en/stable/best-practices.html
- https://docs.dask.org/en/stable/dataframe-best-practices.html
"""

from __future__ import annotations

import multiprocessing
import warnings
from pathlib import Path
from typing import Iterator, Optional, Tuple

import dask
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, wait

from src.utils.logger import get_logger

# Keep PyArrow strings for efficiency (convert to numeric codes for ML later)
# See: https://docs.dask.org/en/stable/dataframe-best-practices.html#avoid-very-large-partitions
dask.config.set({"dataframe.convert-string": False})


LOGGER = get_logger(__name__)


class DataLoader:
	"""Encapsulates train/val/test parquet loading logic using Dask."""

	def __init__(self, config: dict) -> None:
		self.config = config
		self.train_path = Path(config["data"]["train_path"])
		self.test_path = Path(config["data"]["test_path"])
		self._dask_cfg = config.get("dask", {})
		self._client: Optional[Client] = None

	def _ensure_client(self) -> Optional[Client]:
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

	def _read(self, path: Path, filters: Optional[list] = None) -> dd.DataFrame:
		"""Read parquet data with Dask following best practices.
		
		Best Practices:
		- Use filters for partition pruning (very fast!)
		- Keep data in Dask format until needed (avoid compute())
		- Monitor partition sizes (aim for 100-500MB per partition)
		- PyArrow engine is efficient for reading
		
		See: https://docs.dask.org/en/stable/best-practices.html#load-data-with-dask
		"""
		if not path.exists():
			raise FileNotFoundError(f"Dataset path not found: {path}")

		self._ensure_client()
		read_cfg = self._dask_cfg.get("read", {})
		chunksize = read_cfg.get("chunksize")
		read_kwargs = {"chunksize": chunksize} if chunksize else {}

		LOGGER.info("Reading parquet from %s with filters=%s", path, filters)
		ddf = dd.read_parquet(path, filters=filters, engine="pyarrow", **read_kwargs)
		
		# Warn about partition sizes (Best Practice: avoid very large partitions)
		if ddf.npartitions > 0:
			# Estimate partition size (lazy operation, fast)
			memory_per_partition = ddf.memory_usage_per_partition(deep=True).mean()
			try:
				avg_mb = memory_per_partition.compute() / (1024 ** 2)
				if avg_mb > 500:
					LOGGER.warning(
						"⚠️  Large partitions detected: %.1f MB avg. Consider repartitioning for better performance.",
						avg_mb
					)
				elif avg_mb < 10:
					LOGGER.warning(
						"⚠️  Very small partitions detected: %.1f MB avg. Consider using larger partitions.",
						avg_mb
					)
				else:
					LOGGER.info("✅ Partition size looks good: %.1f MB avg", avg_mb)
			except Exception:
				# If computation fails, don't block the read
				pass
				
		return ddf

	def read_path(self, path: str | Path, filters: Optional[list] = None) -> dd.DataFrame:
		"""Public helper to read an arbitrary parquet path with Dask settings."""

		return self._read(Path(path), filters)

	def load_train(
		self,
		validation_split: bool = True,
	) -> Tuple[dd.DataFrame, Optional[dd.DataFrame]]:
		"""Load training data with optional temporal validation split."""

		if validation_split:
			train_filters = [
				("datetime", ">=", self.config["data"]["train_start"]),
				("datetime", "<=", self.config["data"]["train_end"]),
			]
			val_filters = [
				("datetime", ">=", self.config["data"]["val_start"]),
				("datetime", "<=", self.config["data"]["val_end"]),
			]

			train_ddf = self._read(self.train_path, train_filters)
			val_ddf = self._read(self.train_path, val_filters)

			LOGGER.info(
				"Train partitions: %s | Validation partitions: %s",
				train_ddf.npartitions,
				val_ddf.npartitions,
			)
			return train_ddf, val_ddf

		ddf = self._read(self.train_path)
		LOGGER.info("Full train partitions: %s", ddf.npartitions)
		return ddf, None

	def load_test(self) -> dd.DataFrame:
		"""Load official test data window."""

		test_filters = [
			("datetime", ">=", self.config["data"]["test_start"]),
			("datetime", "<=", self.config["data"]["test_end"]),
		]
		ddf = self._read(self.test_path, test_filters)
		LOGGER.info("Test partitions: %s", ddf.npartitions)
		return ddf

	def materialize(
		self,
		ddf: dd.DataFrame,
		*,
		split: str = "train",
		sample_frac: Optional[float] = None,
		max_rows: Optional[int] = None,
		persist: Optional[bool] = None,
		random_state: Optional[int] = None,
	) -> pd.DataFrame:
		"""Convert a Dask DataFrame into pandas with optional sampling.
		
		Best Practice: Avoid calling compute() repeatedly. Build up the computation
		graph and call compute() once at the end.
		
		Note: Function is >40 lines but cohesive - handles materialization pipeline
		with optional sampling, persistence, and validation in a single logical flow.
		
		See: https://docs.dask.org/en/stable/best-practices.html#avoid-calling-compute-repeatedly
		"""
		# Precondition: ddf must be a valid Dask DataFrame
		assert isinstance(ddf, dd.DataFrame), "ddf must be a Dask DataFrame"
		assert len(ddf.columns) > 0, "ddf must have at least one column"

		plan = self._dask_cfg.get("materialization", {}).get(split, {})
		sample_frac = sample_frac if sample_frac is not None else plan.get("sample_frac")
		max_rows = max_rows if max_rows is not None else plan.get("max_rows")
		persist = plan.get("persist", True) if persist is None else persist
		random_state = random_state or self.config.get("training", {}).get("random_state", 42)
		
		# Invariant: sample_frac must be in valid range if specified
		if sample_frac is not None:
			assert 0.0 < sample_frac <= 1.0, f"sample_frac must be in (0, 1], got {sample_frac}"

		# Build the computation graph (all lazy operations)
		result_ddf = ddf
		if sample_frac and sample_frac < 1.0:
			LOGGER.info("Applying sample_frac=%s to %s split", sample_frac, split)
			result_ddf = result_ddf.sample(frac=float(sample_frac), random_state=random_state)

		# Persist is useful for intermediate results used multiple times
		if persist:
			client = self._ensure_client()
			LOGGER.info("Persisting %s split into Dask cache", split)
			result_ddf = result_ddf.persist()
			if client is not None:
				wait(result_ddf)

		# Single compute() call at the end (Best Practice!)
		if max_rows:
			pdf = result_ddf.head(int(max_rows), compute=True)
		else:
			LOGGER.info("Computing %s split to pandas (this may take a while)...", split)
			pdf = result_ddf.compute()

		memory_gb = pdf.memory_usage(deep=True).sum() / 1e9
		LOGGER.info("✅ Materialized %s split: %s rows, %.2f GB", split, len(pdf), memory_gb)
		
		# Postcondition: result must be a valid pandas DataFrame
		assert isinstance(pdf, pd.DataFrame), "Result must be a pandas DataFrame"
		assert len(pdf) > 0 or max_rows == 0, "Result should have rows unless max_rows=0"
		
		# Warn if result is very large
		if memory_gb > 10:
			LOGGER.warning(
				"⚠️  Large dataset in memory (%.2f GB). Consider using sampling or working with Dask directly.",
				memory_gb
			)
		
		return pdf

	@staticmethod
	def iter_batches(ddf: dd.DataFrame, batch_size: int = 100_000) -> Iterator[pd.DataFrame]:
		"""Materialize partitions incrementally, yielding pandas batches.
		
		Best Practice: Avoid calling compute() in a loop. Instead, compute all 
		delayed objects at once and then process them.
		
		This method computes partitions in batches to balance between:
		- Memory efficiency (not loading all data at once)
		- Computation efficiency (not calling compute() excessively)
		
		See: https://docs.dask.org/en/stable/best-practices.html#avoid-calling-compute-repeatedly
		"""
		# Preconditions
		assert isinstance(ddf, dd.DataFrame), "ddf must be a Dask DataFrame"
		assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
		
		delayed_parts = ddf.to_delayed()
		
		# Process in chunks to avoid too many compute() calls
		# Compute multiple partitions at once (e.g., 10 at a time)
		compute_batch_size = 10
		
		cache: list[pd.DataFrame] = []
		cached_rows = 0
		
		for i in range(0, len(delayed_parts), compute_batch_size):
			# Best Practice: Compute multiple delayed objects at once!
			batch = delayed_parts[i:i + compute_batch_size]
			computed_parts = dask.compute(*batch)  # Single compute call for multiple partitions
			
			for part in computed_parts:
				cache.append(part)
				cached_rows += len(part)
				
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
		columns: list[str],
		suffix: str = "_encoded"
	) -> Tuple[dd.DataFrame, dict]:
		"""Encode string columns to numeric for ML models.
		
		Best Practice: String columns are VERY SLOW in Dask. Convert them to
		numeric codes for ML models and numerical operations.
		
		This computes unique values once and applies mappings (lazy operations).
		
		Note: Function is >40 lines but cohesive - computes unique values in a single
		batch and applies all mappings sequentially, which is the correct flow for
		label encoding multiple columns efficiently.
		
		Args:
			ddf: Dask DataFrame
			columns: List of column names to encode
			suffix: Suffix for new encoded columns
			
		Returns:
			Tuple of (DataFrame with new encoded columns, mapping dictionary)
			
		See: https://docs.dask.org/en/stable/dataframe-best-practices.html#avoid-very-large-partitions
		"""
		# Preconditions
		assert isinstance(ddf, dd.DataFrame), "ddf must be a Dask DataFrame"
		assert len(columns) > 0, "columns list must not be empty"
		assert all(col in ddf.columns for col in columns), "All columns must exist in DataFrame"
		
		mappings = {}
		
		# Compute all unique values at once (Best Practice: single compute!)
		LOGGER.info("Computing unique values for %s columns...", len(columns))
		unique_computations = {
			col: ddf[col].unique() for col in columns if col in ddf.columns
		}
		unique_results = dask.compute(unique_computations)[0]
		
		# Create mappings
		for col, unique_vals in unique_results.items():
			mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
			LOGGER.info("  %s: %s unique values", col, len(mappings[col]))
		
		# Apply mappings (lazy operations)
		result_ddf = ddf.copy()
		for col, mapping in mappings.items():
			encoded_col = f"{col}{suffix}"
			result_ddf[encoded_col] = result_ddf[col].map(mapping, na_action='ignore')
			
		return result_ddf, mappings
