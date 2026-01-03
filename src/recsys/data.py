from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.request import urlretrieve
import zipfile
import shutil

import polars as pl


class DataLoader(ABC):
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

    @staticmethod
    def get_stats(df: pl.DataFrame) -> dict[str, float | str]:
        """Calculate basic sparsity statistics for a ratings DataFrame."""

        if not {"user_id", "item_id"}.issubset(df.columns):
            raise ValueError("ratings dataframe must include 'user_id' and 'item_id' columns")

        n_users = df["user_id"].n_unique()
        n_items = df["item_id"].n_unique()
        n_ratings = df.height
        sparsity = 1 - (n_ratings / (n_users * n_items)) if n_users and n_items else 1.0
        return {
            "n_users": n_users,
            "n_items": n_items,
            "n_ratings": n_ratings,
            "sparsity": f"{sparsity:.4%}",
        }


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
    return pl.DataFrame(rows, schema=columns)


@dataclass
class TableSpec:
    """Specification for a single dataset table."""

    filename: str
    read_kwargs: dict
    rename_map: Mapping[str, str] = field(default_factory=dict)
    timestamp_cols: tuple[str, ...] = ()
    preprocess: Callable[[pl.DataFrame], pl.DataFrame] | None = None
    loader: Callable[[Path], pl.DataFrame] | None = None

    def _convert_kwargs(self) -> dict:
        """Translate legacy pandas-style read kwargs to polars-friendly options."""
        kwargs = dict(self.read_kwargs)
        if "sep" in kwargs:
            kwargs["separator"] = kwargs.pop("sep")
        if "names" in kwargs:
            kwargs["new_columns"] = kwargs.pop("names")
        if kwargs.pop("header", None) is None:
            kwargs["has_header"] = False
        kwargs.pop("engine", None)
        return kwargs

    def load(self, base_dir: Path) -> pl.DataFrame:
        """Load a single table with normalization applied."""
        file_path = base_dir / self.filename
        if not file_path.exists():
            raise FileNotFoundError(f"Expected data file not found: {file_path}")

        if self.loader:
            df = self.loader(file_path)
        else:
            kwargs = self._convert_kwargs()
            df = pl.read_csv(file_path, **kwargs)

        if self.rename_map:
            df = df.rename(self.rename_map)
        if self.timestamp_cols:
            df = _parse_timestamps(df, self.timestamp_cols)
        if self.preprocess:
            df = self.preprocess(df)
        return df


@dataclass
class MovieLensConfig:
    """Configuration for a specific MovieLens dataset size."""

    key: str
    archive_url: str
    extract_dirname: str
    tables: dict[str, TableSpec]

    @property
    def archive_name(self) -> str:
        return Path(self.archive_url).name


def _infer_root_folder(archive_path: Path, base_path: Path) -> Path | None:
    """Infer the extracted root folder name from a MovieLens zip archive."""
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        for name in zip_ref.namelist():
            if "/" in name:
                folder = name.split("/")[0]
                root_candidate = base_path / folder
                if root_candidate.exists():
                    return root_candidate
    return None


def download_and_extract(config: MovieLensConfig, data_base_path: Path) -> Path:
    """
    Download and extract a MovieLens archive if it is not already present.

    Args:
        config: Dataset configuration specifying the archive URL and directory names.
        data_base_path: Base directory under which the archive is stored/extracted.

    Returns:
        Path to the dataset extraction directory.
    """
    data_base_path.mkdir(parents=True, exist_ok=True)
    target_dir = data_base_path / config.extract_dirname
    if target_dir.exists() and any(target_dir.iterdir()):
        return target_dir

    archive_path = data_base_path / config.archive_name
    urlretrieve(config.archive_url, archive_path)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(data_base_path)

    extracted_root = _infer_root_folder(archive_path, data_base_path)
    if extracted_root and extracted_root != target_dir:
        extracted_root.rename(target_dir)

    archive_path.unlink(missing_ok=True)
    return target_dir


class MovieLens(DataLoader):
    """Scalable loader for the MovieLens family of datasets."""

    _PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_BASE_PATH = _PROJECT_ROOT / "data" / "movielens"

    def __init__(self, data_base_path: Path | None = None):
        self.data_base_path = data_base_path or self.DATA_BASE_PATH
        self.configs = self._build_configs()

    def _build_configs(self) -> dict[str, MovieLensConfig]:
        """
        Create per-size configs derived from the official dataset READMEs.

        Each config enumerates every table documented for a dataset split so the
        loader can be driven by declarative `TableSpec` objects instead of bespoke
        parsing code.
        """

        genre_cols = [
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]

        return {
            "100k": MovieLensConfig(
                key="100k",
                archive_url="https://files.grouplens.org/datasets/movielens/ml-100k.zip",
                extract_dirname="ml-100k",
                tables={
                    "ratings": TableSpec(
                        filename="u.data",
                        read_kwargs={
                            "sep": "\t",
                            "names": ["user_id", "item_id", "rating", "timestamp"],
                            "header": None,
                            "engine": "python",
                        },
                        timestamp_cols=("timestamp",),
                    ),
                    "movies": TableSpec(
                        filename="u.item",
                        read_kwargs={
                            "sep": "|",
                            "header": None,
                            "encoding": "latin-1",
                            "engine": "python",
                        },
                        preprocess=lambda df: self._assign_columns(
                            df,
                            [
                                "item_id",
                                "title",
                                "release_date",
                                "video_release_date",
                                "imdb_url",
                            ]
                            + genre_cols,
                        ),
                    ),
                    "users": TableSpec(
                        filename="u.user",
                        read_kwargs={
                            "sep": "|",
                            "names": ["user_id", "age", "gender", "occupation", "zip_code"],
                            "header": None,
                            "engine": "python",
                        },
                    ),
                    # Auxiliary metadata files exposed by the README
                    "genres": TableSpec(
                        filename="u.genre",
                        read_kwargs={"sep": "|", "names": ["genre", "genre_id"], "header": None},
                    ),
                    "occupations": TableSpec(
                        filename="u.occupation",
                        read_kwargs={"header": None, "names": ["occupation"]},
                    ),
                },
            ),
            "1m": MovieLensConfig(
                key="1m",
                archive_url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
                extract_dirname="ml-1m",
                tables={
                    "ratings": TableSpec(
                        filename="ratings.dat",
                        read_kwargs={"names": ["user_id", "item_id", "rating", "timestamp"], "header": None},
                        loader=lambda path: _read_multichar_separated(
                            path, "::", ["user_id", "item_id", "rating", "timestamp"]
                        ),
                        timestamp_cols=("timestamp",),
                    ),
                    "movies": TableSpec(
                        filename="movies.dat",
                        read_kwargs={"names": ["item_id", "title", "genres"], "header": None},
                        loader=lambda path: _read_multichar_separated(
                            path, "::", ["item_id", "title", "genres"], encoding="latin-1"
                        ),
                    ),
                    "users": TableSpec(
                        filename="users.dat",
                        read_kwargs={"names": ["user_id", "gender", "age", "occupation", "zip_code"], "header": None},
                        loader=lambda path: _read_multichar_separated(
                            path, "::", ["user_id", "gender", "age", "occupation", "zip_code"]
                        ),
                    ),
                },
            ),
            "10m": MovieLensConfig(
                key="10m",
                archive_url="https://files.grouplens.org/datasets/movielens/ml-10m.zip",
                extract_dirname="ml-10m",
                tables={
                    "ratings": TableSpec(
                        filename="ratings.dat",
                        read_kwargs={"names": ["user_id", "item_id", "rating", "timestamp"], "header": None},
                        loader=lambda path: _read_multichar_separated(
                            path, "::", ["user_id", "item_id", "rating", "timestamp"]
                        ),
                        timestamp_cols=("timestamp",),
                    ),
                    "movies": TableSpec(
                        filename="movies.dat",
                        read_kwargs={"names": ["item_id", "title", "genres"], "header": None},
                        loader=lambda path: _read_multichar_separated(
                            path, "::", ["item_id", "title", "genres"], encoding="latin-1"
                        ),
                    ),
                    "tags": TableSpec(
                        filename="tags.dat",
                        read_kwargs={"names": ["user_id", "item_id", "tag", "timestamp"], "header": None},
                        loader=lambda path: _read_multichar_separated(
                            path, "::", ["user_id", "item_id", "tag", "timestamp"]
                        ),
                        timestamp_cols=("timestamp",),
                    ),
                },
            ),
            "20m": MovieLensConfig(
                key="20m",
                archive_url="https://files.grouplens.org/datasets/movielens/ml-20m.zip",
                extract_dirname="ml-20m",
                tables={
                    "ratings": TableSpec(
                        filename="ratings.csv",
                        read_kwargs={"engine": "python"},
                        rename_map={"userId": "user_id", "movieId": "item_id"},
                        timestamp_cols=("timestamp",),
                    ),
                    "movies": TableSpec(
                        filename="movies.csv",
                        read_kwargs={"engine": "python"},
                        rename_map={"movieId": "item_id"},
                    ),
                    "tags": TableSpec(
                        filename="tags.csv",
                        read_kwargs={"engine": "python"},
                        rename_map={"userId": "user_id", "movieId": "item_id"},
                        timestamp_cols=("timestamp",),
                    ),
                    "genome_scores": TableSpec(
                        filename="genome-scores.csv",
                        read_kwargs={"engine": "python"},
                        rename_map={"movieId": "item_id"},
                    ),
                    "genome_tags": TableSpec(
                        filename="genome-tags.csv",
                        read_kwargs={"engine": "python"},
                    ),
                    "links": TableSpec(
                        filename="links.csv",
                        read_kwargs={"engine": "python"},
                        rename_map={"movieId": "item_id"},
                    ),
                },
            ),
            "32m": MovieLensConfig(
                key="32m",
                archive_url="https://files.grouplens.org/datasets/movielens/ml-32m.zip",
                extract_dirname="ml-32m",
                tables={
                    "ratings": TableSpec(
                        filename="ratings.csv",
                        read_kwargs={"engine": "python"},
                        rename_map={"userId": "user_id", "movieId": "item_id"},
                        timestamp_cols=("timestamp",),
                    ),
                    "movies": TableSpec(
                        filename="movies.csv",
                        read_kwargs={"engine": "python"},
                        rename_map={"movieId": "item_id"},
                    ),
                    "tags": TableSpec(
                        filename="tags.csv",
                        read_kwargs={"engine": "python"},
                        rename_map={"userId": "user_id", "movieId": "item_id"},
                        timestamp_cols=("timestamp",),
                    ),
                    "links": TableSpec(
                        filename="links.csv",
                        read_kwargs={"engine": "python"},
                        rename_map={"movieId": "item_id"},
                    ),
                },
            ),
        }

    @staticmethod
    def _assign_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        df.columns = columns
        return df

    @property
    def available_datasets(self) -> list[str]:
        """Return sorted list of supported MovieLens dataset keys."""
        return sorted(self.configs)

    def load(
        self,
        size: str = "100k",
        tables: Iterable[str] | None = None,
        download: bool = True,
    ) -> dict[str, pl.DataFrame]:
        """
        Load one or more MovieLens tables for a specific dataset size.

        Args:
            size: Dataset key ("100k", "1m", "10m", "20m", "32m").
            tables: Iterable of table names to load. If None, every table from
                the corresponding README is loaded (e.g., `genres`/`occupations`
                for 100k, `genome_*` for 20m, etc.).
            download: Whether to fetch + extract the dataset automatically if
                it is missing locally.

        Returns:
            Mapping of table name -> pandas DataFrame.
        """

        size_key = size.lower()
        if size_key not in self.configs:
            raise ValueError(f"Unsupported dataset size: {size}")

        config = self.configs[size_key]
        if download:
            base_dir = download_and_extract(config, self.data_base_path)
        else:
            base_dir = self.data_base_path / config.extract_dirname
            if not base_dir.exists():
                raise FileNotFoundError(
                    f"Dataset {size_key} not found at {base_dir}. Set download=True to fetch it."
                )

        requested_tables = set(tables) if tables else set(config.tables)
        unknown = requested_tables - set(config.tables)
        if unknown:
            raise ValueError(f"Unknown tables for {size_key}: {sorted(unknown)}")

        return {name: config.tables[name].load(base_dir) for name in requested_tables}

    def save(
        self,
        size: str = "100k",
        tables: Iterable[str] | None = None,
        dest: str | Path | None = None,
        file_format: str = "parquet",
        raw: bool = False,
        download: bool = True,
    ) -> Path:
        """
        Persist MovieLens data to a target directory (local path or a mounted remote
        location such as Google Drive).

        Args:
            size: Dataset key to save.
            tables: Subset of tables to save when `raw` is False. Defaults to all.
            dest: Destination directory. Useful for saving to alternative storage
                (e.g., a Google Drive mount path). Defaults to a `<size>_export`
                folder inside the base data directory.
            file_format: File format for processed tables ("parquet" or "csv").
            raw: When True, copy the raw extracted files instead of processed tables.
            download: Whether to download/extract before saving when missing locally.

        Returns:
            Path to the directory containing the saved data.
        """

        size_key = size.lower()
        config = self.configs.get(size_key)
        if not config:
            raise ValueError(f"Unsupported dataset size: {size}")

        if download:
            base_dir = download_and_extract(config, self.data_base_path)
        else:
            base_dir = self.data_base_path / config.extract_dirname
            if not base_dir.exists():
                raise FileNotFoundError(
                    f"Dataset {size_key} not found at {base_dir}. Set download=True to fetch it."
                )

        dest_path = Path(dest).expanduser() if dest else self.data_base_path / f"{size_key}_export"
        dest_path.mkdir(parents=True, exist_ok=True)

        if raw:
            shutil.copytree(base_dir, dest_path, dirs_exist_ok=True)
            return dest_path

        fmt = file_format.lower()
        if fmt not in {"parquet", "csv"}:
            raise ValueError("file_format must be 'parquet' or 'csv'")

        data = self.load(size=size_key, tables=tables, download=False)
        for name, df in data.items():
            out_path = dest_path / f"{name}.{fmt}"
            if fmt == "parquet":
                df.write_parquet(out_path)
            else:
                df.write_csv(out_path)

        return dest_path

    # Convenience utilities for experimentation ---------------------------------
    @staticmethod
    def to_implicit(ratings: pl.DataFrame, threshold: float = 0.0) -> pl.DataFrame:
        """Convert explicit ratings to implicit feedback labels."""

        if "rating" not in ratings.columns:
            raise ValueError("ratings dataframe must contain a 'rating' column")
        return ratings.with_columns((pl.col("rating") > threshold).cast(pl.Int8).alias("rating"))

    @staticmethod
    def filter_cold_start(
        ratings: pl.DataFrame,
        min_user_interactions: int = 1,
        min_item_interactions: int = 1,
    ) -> pl.DataFrame:
        """Drop users/items with too few interactions for stable evaluation."""

        user_counts = ratings.group_by("user_id").count().rename({"count": "user_interactions"})
        item_counts = ratings.group_by("item_id").count().rename({"count": "item_interactions"})

        enriched = (
            ratings.join(user_counts, on="user_id", how="left")
            .join(item_counts, on="item_id", how="left")
        )
        filtered = enriched.filter(
            (pl.col("user_interactions") >= min_user_interactions)
            & (pl.col("item_interactions") >= min_item_interactions)
        )
        return filtered.drop(["user_interactions", "item_interactions"])

    @staticmethod
    def chronological_split(
        ratings: pl.DataFrame, test_ratio: float = 0.2, timestamp_col: str = "timestamp"
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Split ratings by timestamp for temporal evaluation."""

        if timestamp_col not in ratings.columns:
            raise ValueError(f"ratings dataframe must contain '{timestamp_col}' for chronological split")

        sorted_ratings = ratings.sort(timestamp_col)
        split_idx = int(sorted_ratings.height * (1 - test_ratio))
        train = sorted_ratings.slice(0, split_idx)
        test = sorted_ratings.slice(split_idx)
        return train, test
