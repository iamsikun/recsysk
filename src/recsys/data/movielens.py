from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from recsys.data.utils import (
    BaseDataLoader,
    TableSpec,
    _read_multichar_separated,
)
from recsys.schemas.features import FeatureSpec, FeatureType
from recsys.utils import DATASET_REGISTRY
from recsys.schemas.builder import build_feature_specs


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
                    # Auxiliary metadata files exposed by the README
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
        """
        Load one or more MovieLens tables for a specific dataset size.

        Args:
            size: Dataset key ("100k", "1m", "10m", "20m", "32m").
            tables: Iterable of table names to load. If None, every table from
                the corresponding README is loaded (e.g., `genres`/`occupations`
                for 100k, `genome_*` for 20m, etc.).

        Returns:
            Mapping of table name -> polars DataFrame.
        """

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


class MovieLensDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        # We assume X and y are already encoded tensors or mmap arrays
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MovieLensDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for MovieLens datasets.

    This class organizes all aspects of data handling:
    - Data loading from disk
    - Feature encoding (categorical encoding)
    - Dataset creation
    - Train/validation split
    - DataLoader creation

    The DataModule follows Lightning's standard lifecycle:
    1. prepare_data(): Download/prepare data (optional, can be done separately)
    2. setup(): Load data, create encoders, prepare datasets
    3. train_dataloader(): Return training DataLoader
    4. val_dataloader(): Return validation DataLoader

    Example:
        >>> from pathlib import Path
        >>> datamodule = MovieLensDataModule(
        ...     data_base_path=Path("../datasets"),
        ...     dataset_size="20m",
        ...     batch_size=1024,
        ...     train_split=0.8,
        ...     rating_threshold=3.5,
        ...     num_workers=4,
        ... )
        >>> datamodule.setup()
        >>> train_loader = datamodule.train_dataloader()
        >>> val_loader = datamodule.val_dataloader()
    """

    def __init__(
        self,
        data_base_path: Path,
        dataset_size: str = "20m",
        batch_size: int = 1024,
        train_split: float = 0.8,
        rating_threshold: float = 3.5,
        num_workers: int = 0,
        pin_memory: bool = False,
        features: list[FeatureSpec] = None,
    ):
        """
        Initialize the MovieLens DataModule.

        Args:
            data_base_path: Base path to the datasets directory.
                The MovieLens data should be extracted under this path.
            dataset_size: Size of the MovieLens dataset to use.
                Options: "100k", "1m", "10m", "20m", "32m"
            batch_size: Batch size for DataLoaders.
            train_split: Proportion of data to use for training (rest goes to validation).
                Should be between 0.0 and 1.0.
            rating_threshold: Threshold for converting ratings to binary labels.
                Ratings >= threshold become 1 (positive), otherwise 0 (negative).
            num_workers: Number of worker processes for data loading.
                0 means data loading happens in the main process.
            pin_memory: Whether to pin memory in DataLoader for faster GPU transfer.
            features: List of FeatureSpec objects defining the features to use.
        """
        super().__init__()
        self.data_base_path = Path(data_base_path)
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.train_split = train_split
        self.rating_threshold = rating_threshold
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.features = features

        # These will be set in setup()
        self.ml_loader: MovieLens | None = None
        self.fields_info: dict[str, dict[str, Any]] | None = None
        self.train_dataset: MovieLensDataset | None = None
        self.val_dataset: MovieLensDataset | None = None

        # Default features
        if features is None:
            features = [
                FeatureSpec(
                    name="user_id", source_name="user_id", type=FeatureType.CATEGORICAL
                ),
                FeatureSpec(
                    name="item_id", source_name="item_id", type=FeatureType.CATEGORICAL
                ),
            ]

        self.feature_map = {}

    def prepare_data(self) -> None:
        """
        Download or prepare data if needed.

        This method is called only on the main process (rank 0) in distributed settings.
        It's optional - you can also download/prepare data separately.

        Note: The actual data loading happens in setup(), which is called on all processes.
        """
        # Data preparation can be done here if needed
        # For now, we assume data is already downloaded and extracted
        pass

    def setup(self, stage: str | None = None) -> None:
        # Initialize the MovieLens loader
        self.ml_loader = MovieLens(self.data_base_path)

        # Load ratings data
        data: dict = self.ml_loader.load(self.dataset_size, tables=["ratings"])
        df: pl.DataFrame = data["ratings"]

        # Global preprocessing (label generation)
        df = df.with_columns(
            (pl.col("rating").cast(pl.Float32) >= self.rating_threshold)
            .cast(pl.Int32)
            .alias("label")
        )

        # Dynamic preprocessing loop
        processed_cols = []
        for spec in self.features:
            col_name = spec.source_name

            if spec.type == FeatureType.CATEGORICAL:
                # Encode categorical feature using polars categorical type
                # Cast to string first, then to categorical (required for numeric types)
                # Cast to int64 for PyTorch embedding layer compatibility
                df = df.with_columns(
                    pl.col(col_name)
                    .cast(pl.Utf8)
                    .cast(pl.Categorical)
                    .to_physical()
                    .cast(pl.Int64)
                    .alias(spec.name)
                )

                # record meta data
                spec.vocab_size = (
                    df[spec.name].max() + 1
                )  # +1 for potential OOV or 0-indexing
                self.feature_map[spec.name] = spec.vocab_size
            elif spec.type == FeatureType.NUMERIC:
                df = df.with_columns(pl.col(col_name).cast(pl.Float32).alias(spec.name))

                self.feature_map[spec.name] = 1

            processed_cols.append(spec.name)

        # Extract label before selecting only feature columns
        label_tensor = torch.from_numpy(
            df["label"].to_numpy().astype("float32")
        ).unsqueeze(-1)

        # Select only feature columns for data tensor
        df = df.select(processed_cols)

        # create tensor
        # Note: categorical features are already cast to int64 in Polars above
        # For mixed types, we'd need to handle conversion per column type
        data_tensor = torch.from_numpy(df.to_numpy())

        # create dataset
        self.full_dataset = MovieLensDataset(data_tensor, label_tensor)

        # split
        train_size = int(len(self.full_dataset) * self.train_split)
        val_size = len(self.full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset, [train_size, val_size]
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create and return the training DataLoader.

        Returns:
            DataLoader for training data with shuffling enabled.
        """
        if self.train_dataset is None:
            raise RuntimeError("Must call setup() before train_dataloader()")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create and return the validation DataLoader.

        Returns:
            DataLoader for validation data with shuffling disabled.
        """
        if self.val_dataset is None:
            raise RuntimeError("Must call setup() before val_dataloader()")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


# register dataset
@DATASET_REGISTRY.register("movielens")
def build_movielens_datamodule(
    data_base_path: Path,
    batch_size: int,
    features: list[FeatureSpec],
    dataset_size: str = "20m",
    train_split: float = 0.8,
    rating_threshold: float = 3.5,
    num_workers: int = 0,
    pin_memory: bool = False,
    **kwargs: Any,
) -> MovieLensDataModule:
    """
    Build a MovieLensDataModule.

    Args:
        data_base_path: Base path to the datasets directory.
        batch_size: Batch size for DataLoaders.
        features: List of FeatureSpec objects defining the features to use.
        dataset_size: Size of the MovieLens dataset to use.
        train_split: Proportion of data to use for training (rest goes to validation).
        rating_threshold: Threshold for converting ratings to binary labels.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory in DataLoader for faster GPU transfer.
    """
    # Make sure the data_base_path is a valid path
    data_base_path = Path(data_base_path).resolve()
    return MovieLensDataModule(
        data_base_path,
        dataset_size,
        batch_size,
        train_split,
        rating_threshold,
        num_workers,
        pin_memory,
        features=build_feature_specs(features),
        **kwargs,
    )
