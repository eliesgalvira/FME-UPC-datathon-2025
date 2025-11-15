"""Dask-powered parquet loading utilities."""

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
		client_cfg = self._dask_cfg.get("client", {})
		if not client_cfg.get("enabled", False):
			return None
		if self._client is not None:
			return self._client

		n_workers = client_cfg.get("n_workers")
		if n_workers is None:
			cpu_count = max(1, multiprocessing.cpu_count() - 1)
			n_workers = max(1, cpu_count)
		threads = client_cfg.get("threads_per_worker", 2)
		memory_limit = client_cfg.get("memory_limit", "4GB")
		self._client = Client(
			n_workers=n_workers,
			threads_per_worker=threads,
			memory_limit=memory_limit,
		)
		LOGGER.info("Started Dask client (%s workers, dashboard %s)", n_workers, self._client.dashboard_link)
		return self._client

	def close(self) -> None:
		if self._client is not None:
			LOGGER.info("Closing Dask client")
			self._client.close()
			self._client = None

	def _read(self, path: Path, filters: Optional[list] = None) -> dd.DataFrame:
		if not path.exists():
			raise FileNotFoundError(f"Dataset path not found: {path}")

		self._ensure_client()
		read_cfg = self._dask_cfg.get("read", {})
		chunksize = read_cfg.get("chunksize")
		read_kwargs = {"chunksize": chunksize} if chunksize else {}

		LOGGER.info("Reading parquet from %s", path)
		return dd.read_parquet(path, filters=filters, engine="pyarrow", **read_kwargs)

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
		"""Convert a Dask DataFrame into pandas with optional sampling."""

		plan = self._dask_cfg.get("materialization", {}).get(split, {})
		sample_frac = sample_frac if sample_frac is not None else plan.get("sample_frac")
		max_rows = max_rows if max_rows is not None else plan.get("max_rows")
		persist = plan.get("persist", True) if persist is None else persist
		random_state = random_state or self.config.get("training", {}).get("random_state", 42)

		result_ddf = ddf
		if sample_frac and sample_frac < 1.0:
			LOGGER.info("Applying sample_frac=%s to %s split", sample_frac, split)
			result_ddf = result_ddf.sample(frac=float(sample_frac), random_state=random_state)

		if persist:
			client = self._ensure_client()
			LOGGER.info("Persisting %s split into Dask cache", split)
			result_ddf = result_ddf.persist()
			if client is not None:
				wait(result_ddf)

		if max_rows:
			pdf = result_ddf.head(int(max_rows), compute=True)
		else:
			pdf = result_ddf.compute()

		memory_gb = pdf.memory_usage(deep=True).sum() / 1e9
		LOGGER.info("Materialized %s split with %s rows (%.2f GB)", split, len(pdf), memory_gb)
		return pdf

	@staticmethod
	def iter_batches(ddf: dd.DataFrame, batch_size: int = 100_000) -> Iterator[pd.DataFrame]:
		"""Materialize partitions incrementally, yielding pandas batches."""

		cache: list[pd.DataFrame] = []
		cached_rows = 0
		for delayed_part in ddf.to_delayed():
			part = delayed_part.compute()
			cache.append(part)
			cached_rows += len(part)
			if cached_rows >= batch_size:
				yield pd.concat(cache, ignore_index=True)
				cache.clear()
				cached_rows = 0

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
