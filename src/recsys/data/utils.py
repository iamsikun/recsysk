from __future__ import annotations

import os 
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix


def get_ratings_stats(ratings: pl.DataFrame) -> dict[str, int | float]:
    """Calculate basic sparsity statistics for a ratings DataFrame."""

    if not {"user_id", "item_id"}.issubset(ratings.columns):
        raise ValueError("ratings dataframe must include 'user_id' and 'item_id' columns")

    n_users = ratings["user_id"].n_unique()
    n_items = ratings["item_id"].n_unique()
    n_ratings = ratings.height
    sparsity = 1 - (n_ratings / (n_users * n_items)) if n_users and n_items else 1.0
    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": n_ratings,
        "sparsity": sparsity,
    }



def convert_ratings_table_to_matrix(ratings: pl.DataFrame) -> coo_matrix:
    """Convert a ratings table to a sparse matrix (users x items)."""
    # Map user and item ids to contiguous indices
    user_ids, user_idx = np.unique(ratings["user_id"].to_numpy(), return_inverse=True)
    item_ids, item_idx = np.unique(ratings["item_id"].to_numpy(), return_inverse=True)
    data = ratings["rating"].cast(pl.Float32).to_numpy()
    num_users = user_ids.shape[0]
    num_items = item_ids.shape[0]
    matrix = coo_matrix(
        (data, (user_idx, item_idx)),
        shape=(num_users, num_items)
    )
    return matrix


def _parse_timestamps(df: pl.DataFrame, timestamp_cols: Iterable[str]) -> pl.DataFrame:
    """Cast integer timestamp columns to polars datetime (seconds)."""
    expressions = []
    for col in timestamp_cols:
        if col in df.columns:
            expressions.append(pl.from_epoch(pl.col(col).cast(pl.Int64, strict=False), time_unit="s").alias(col))
    return df.with_columns(expressions) if expressions else df


def _read_multichar_separated(
    file_path: Path, separator: str, columns: list[str], encoding: str | None = None
) -> pl.DataFrame:
    """
    Read a text file that uses a multi-character delimiter (e.g., '::') into a Polars DataFrame.
    """
    with open(file_path, "r", encoding=encoding or "utf-8") as f:
        rows = [line.rstrip("\n").split(separator) for line in f]
    return pl.DataFrame(rows, schema=columns, orient="row")


### DataLoader class ###

class BaseDataLoader(ABC):
    """
    Minimal abstract interface for dataset loaders.

    Every concrete loader should expose a `load` entrypoint that returns the
    relevant tables for a dataset. Additional utilities can be added as static
    methods to avoid growing separate helper modules.
    """

    @abstractmethod
    def load(self, *args, **kwargs):
        """
        Load one or more tables for a dataset.

        Subclasses are expected to document their supported parameters and the
        structure of the returned data.
        """
        msg = "Subclasses must implement the load method"
        raise NotImplementedError(msg)


@dataclass
class TableSpec:
    """Specification for a single dataset table."""

    filename: str  # name of the file to load
    read_kwargs: dict  # kwargs for the polars read_csv function
    rename_map: Mapping[str, str] = field(default_factory=dict)  # map of old column names to new column names
    timestamp_cols: tuple[str, ...] = ()  # columns to parse as timestamps
    preprocess: Callable[[pl.DataFrame], pl.DataFrame] | None = None  # function to apply to the dataframe after loading
    loader: Callable[[Path], pl.DataFrame] | None = None  # function to load the dataframe from a file

    def _convert_kwargs(self) -> dict:
        """Translate legacy pandas-style read kwargs to polars-friendly options."""
        kwargs = dict(self.read_kwargs)
        if "sep" in kwargs:
            kwargs["separator"] = kwargs.pop("sep")
        if "names" in kwargs:
            kwargs["new_columns"] = kwargs.pop("names")
        # Handle header: pandas uses header=None for no header, header=0 (default) for has header
        # polars uses has_header=True (default) for has header, has_header=False for no header
        header_val = kwargs.pop("header", 0)  # default to 0 (has header, pandas default)
        if header_val is None:
            kwargs["has_header"] = False
        elif header_val == 0:
            kwargs["has_header"] = True
        # For other header values, keep polars default (has_header=True)
        kwargs.pop("engine", None)
        return kwargs

    def load(self, base_dir: Path) -> pl.DataFrame:
        """Load a single table with normalization applied."""
        file_path = base_dir / self.filename
        if not file_path.exists():
            raise FileNotFoundError(f"Expected data file not found: {file_path}")

        if self.loader:  # use the loader function if provided
            df = self.loader(file_path)
        else:
            kwargs = self._convert_kwargs()
            df = pl.read_csv(file_path, **kwargs)

        # rename columns if provided
        if self.rename_map:
            df = df.rename(self.rename_map)

        # parse timestamps if provided
        if self.timestamp_cols:
            df = _parse_timestamps(df, self.timestamp_cols)

        # apply preprocess function if provided
        if self.preprocess:
            df = self.preprocess(df)
        return df

