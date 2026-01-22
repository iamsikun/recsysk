from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from recsys.data.utils import BaseDataLoader, TableSpec, _read_multichar_separated


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


class MovieLens(BaseDataLoader):
    """Scalable loader for the MovieLens family of datasets."""

    def __init__(self, data_base_path: Path):
        self.data_base_path = data_base_path
        self.configs = self._build_configs()

    def _build_configs(self) -> dict[str, MovieLensConfig]:
        """
        Create per-size configs derived from the official dataset READMEs.
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
                            "names": [
                                "user_id",
                                "age",
                                "gender",
                                "occupation",
                                "zip_code",
                            ],
                            "header": None,
                            "engine": "python",
                        },
                    ),
                    "genres": TableSpec(
                        filename="u.genre",
                        read_kwargs={
                            "sep": "|",
                            "names": ["genre", "genre_id"],
                            "header": None,
                        },
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
                        read_kwargs={
                            "names": ["user_id", "item_id", "rating", "timestamp"],
                            "header": None,
                        },
                        loader=lambda path: _read_multichar_separated(
                            path, "::", ["user_id", "item_id", "rating", "timestamp"]
                        ),
                        timestamp_cols=("timestamp",),
                    ),
                    "movies": TableSpec(
                        filename="movies.dat",
                        read_kwargs={
                            "names": ["item_id", "title", "genres"],
                            "header": None,
                        },
                        loader=lambda path: _read_multichar_separated(
                            path,
                            "::",
                            ["item_id", "title", "genres"],
                            encoding="latin-1",
                        ),
                    ),
                    "users": TableSpec(
                        filename="users.dat",
                        read_kwargs={
                            "names": [
                                "user_id",
                                "gender",
                                "age",
                                "occupation",
                                "zip_code",
                            ],
                            "header": None,
                        },
                        loader=lambda path: _read_multichar_separated(
                            path,
                            "::",
                            ["user_id", "gender", "age", "occupation", "zip_code"],
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
                        read_kwargs={
                            "names": ["user_id", "item_id", "rating", "timestamp"],
                            "header": None,
                        },
                        loader=lambda path: _read_multichar_separated(
                            path, "::", ["user_id", "item_id", "rating", "timestamp"]
                        ),
                        timestamp_cols=("timestamp",),
                    ),
                    "movies": TableSpec(
                        filename="movies.dat",
                        read_kwargs={
                            "names": ["item_id", "title", "genres"],
                            "header": None,
                        },
                        loader=lambda path: _read_multichar_separated(
                            path,
                            "::",
                            ["item_id", "title", "genres"],
                            encoding="latin-1",
                        ),
                    ),
                    "tags": TableSpec(
                        filename="tags.dat",
                        read_kwargs={
                            "names": ["user_id", "item_id", "tag", "timestamp"],
                            "header": None,
                        },
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
    ) -> dict[str, pl.DataFrame]:
        """Load one or more MovieLens tables for a specific dataset size."""
        size_key = size.lower()
        if size_key not in self.configs:
            raise ValueError(f"Unsupported dataset size: {size}")

        config = self.configs[size_key]
        base_dir = self.data_base_path / config.extract_dirname

        requested_tables = set(tables) if tables else set(config.tables)
        unknown = requested_tables - set(config.tables)
        if unknown:
            raise ValueError(f"Unknown tables for {size_key}: {sorted(unknown)}")

        return {name: config.tables[name].load(base_dir) for name in requested_tables}
