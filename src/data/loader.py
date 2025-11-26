"""Dask-powered parquet loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import dask
import dask.dataframe as dd
import pandas as pd

from src.utils.logger import get_logger

dask.config.set({"dataframe.convert-string": False})


LOGGER = get_logger(__name__)


class DataLoader:
	"""Encapsulates train/val/test parquet loading logic using Dask."""

	def __init__(self, config: dict) -> None:
		self.config = config
		self.train_path = Path(config["data"]["train_path"])
		self.test_path = Path(config["data"]["test_path"])

	def _read(self, path: Path, filters: Optional[list] = None) -> dd.DataFrame:
		if not path.exists():
			raise FileNotFoundError(f"Dataset path not found: {path}")

		LOGGER.info("Reading parquet from %s", path)
		return dd.read_parquet(path, filters=filters, engine="pyarrow")

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

	@staticmethod
	def compute_in_batches(
		ddf: dd.DataFrame, batch_size: int = 100_000
	) -> Iterator[pd.DataFrame]:
		"""Materialize a Dask dataframe in manageable pandas batches."""

		cache: list[pd.DataFrame] = []
		cached_rows = 0
		for partition_idx in range(ddf.npartitions):
			part = ddf.get_partition(partition_idx).compute()
			cache.append(part)
			cached_rows += len(part)

			if cached_rows >= batch_size:
				yield pd.concat(cache, ignore_index=True)
				cache.clear()
				cached_rows = 0

		if cache:
			yield pd.concat(cache, ignore_index=True)
