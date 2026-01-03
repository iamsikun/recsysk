from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import pandas as pd


class DataLoader(ABC):
    """Abstract Base Class for Data Loading"""
    @abstractmethod
    def load_data(self):
        pass
    
    def get_stats(self, df):
        """Calculate dataset statistics."""
        n_users = df['user_id'].nunique()
        n_items = df['item_id'].nunique()
        n_ratings = len(df)
        sparsity = 1 - (n_ratings / (n_users * n_items))
        return {
            'n_users': n_users, 
            'n_items': n_items, 
            'n_ratings': n_ratings, 
            'sparsity': f"{sparsity:.4%}"
        }


class MovieLensDownloader:
    """
    Unified downloader for all MovieLens datasets.
    Handles downloading and extracting zip files.
    """
    DATA_URL_DICT: dict[str, str] = {
        "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
        "20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
    }
    
    # Get project root (two levels up from this file: src/recsys/data.py -> root)
    _PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_BASE_PATH = _PROJECT_ROOT / "data" / "movielens"

    def download_and_extract(self, size: str):
        """
        Downloads and extracts the MovieLens dataset of the given size if not already present.
        
        Args:
            size: Dataset size ('100k', '1m', '10m', '20m', '32m')
        """
        size_lc = size.lower()
        if size_lc not in self.DATA_URL_DICT:
            raise ValueError(f"Unsupported dataset size: {size}")
        
        extract_dir = self.DATA_BASE_PATH / f"ml-{size_lc}"
        
        # Check if extract directory exists and is not empty
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"Data already extracted at: {extract_dir}")
            return

        url = self.DATA_URL_DICT[size_lc]
        archive_name = f"ml-{size_lc}.zip"
        full_archive_path = self.DATA_BASE_PATH / archive_name

        # Create data directory if it doesn't exist
        self.DATA_BASE_PATH.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading MovieLens {size} from {url}...")
        urlretrieve(url, full_archive_path)
        
        print("Extracting zip...")
        with zipfile.ZipFile(full_archive_path, "r") as zip_ref:
            # Get the root directory name from the zip file
            zip_names = zip_ref.namelist()
            if not zip_names:
                raise ValueError(f"Empty zip file: {archive_name}")
            
            # Find the root directory (first path component)
            root_dir_name = None
            for name in zip_names:
                if '/' in name:
                    root_dir_name = name.split('/')[0]
                    break
            
            # Extract the zip
            zip_ref.extractall(self.DATA_BASE_PATH)
            
            # Rename the extracted directory to the expected name
            if root_dir_name:
                extracted_dir = self.DATA_BASE_PATH / root_dir_name
                if extracted_dir.exists() and extracted_dir != extract_dir:
                    extracted_dir.rename(extract_dir)
        
        # Clean up archive
        full_archive_path.unlink()
        print(f"Download and extraction complete for MovieLens {size}.")


class MovieLensProcessor(ABC):
    """Abstract base class for MovieLens dataset processors."""
    
    def __init__(self, data_base_path: Path):
        self.data_base_path = data_base_path
    
    @abstractmethod
    def load_ratings(self) -> pd.DataFrame:
        """Load ratings data into a DataFrame."""
        pass
    
    @abstractmethod
    def get_extract_dir(self) -> Path:
        """Get the extraction directory for this dataset."""
        pass
    
    @abstractmethod
    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all available data for this dataset into a dictionary."""
        pass


class MovieLens100kProcessor(MovieLensProcessor):
    """Processor for MovieLens 100k dataset."""
    
    def get_extract_dir(self) -> Path:
        return self.data_base_path / "ml-100k"
    
    def load_ratings(self) -> pd.DataFrame:
        """Load ratings from u.data file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "u.data"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {file_path}")
        
        df = pd.read_csv(
            file_path,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            header=None,
            engine='python'
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        return df
    
    def load_movies(self) -> pd.DataFrame:
        """Load movie information from u.item file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "u.item"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Movies file not found: {file_path}")
        
        # u.item format: movie id | title | release date | video release date | IMDb URL | genres (19 binary)
        df = pd.read_csv(
            file_path,
            sep='|',
            encoding='latin-1',
            header=None,
            engine='python'
        )
        # First 5 columns: id, title, release_date, video_release_date, imdb_url
        # Remaining 19 columns are genres
        genre_cols = [
            'unknown', 'Action', 'Adventure', 'Animation', "Children's",
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
            'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        df.columns = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols
        return df
    
    def load_users(self) -> pd.DataFrame:
        """Load user information from u.user file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "u.user"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Users file not found: {file_path}")
        
        df = pd.read_csv(
            file_path,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            header=None,
            engine='python'
        )
        return df
    
    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all available data for the 100k dataset."""
        return {
            'ratings': self.load_ratings(),
            'movies': self.load_movies(),
            'users': self.load_users(),
        }


class MovieLens1mProcessor(MovieLensProcessor):
    """Processor for MovieLens 1m dataset."""
    
    def get_extract_dir(self) -> Path:
        return self.data_base_path / "ml-1m"
    
    def load_ratings(self) -> pd.DataFrame:
        """Load ratings from ratings.dat file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "ratings.dat"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {file_path}")
        
        df = pd.read_csv(
            file_path,
            sep='::',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            header=None,
            engine='python'
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        return df
    
    def load_movies(self) -> pd.DataFrame:
        """Load movie information from movies.dat file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "movies.dat"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Movies file not found: {file_path}")
        
        df = pd.read_csv(
            file_path,
            sep='::',
            names=['item_id', 'title', 'genres'],
            header=None,
            encoding='latin-1',
            engine='python'
        )
        return df
    
    def load_users(self) -> pd.DataFrame:
        """Load user information from users.dat file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "users.dat"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Users file not found: {file_path}")
        
        df = pd.read_csv(
            file_path,
            sep='::',
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
            header=None,
            engine='python'
        )
        return df
    
    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all available data for the 1m dataset."""
        return {
            'ratings': self.load_ratings(),
            'movies': self.load_movies(),
            'users': self.load_users(),
        }


class MovieLens10mProcessor(MovieLensProcessor):
    """Processor for MovieLens 10m dataset."""
    
    def get_extract_dir(self) -> Path:
        return self.data_base_path / "ml-10m"
    
    def load_ratings(self) -> pd.DataFrame:
        """Load ratings from ratings.dat file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "ratings.dat"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {file_path}")
        
        df = pd.read_csv(
            file_path,
            sep='::',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            header=None,
            engine='python'
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        return df
    
    def load_movies(self) -> pd.DataFrame:
        """Load movie information from movies.dat file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "movies.dat"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Movies file not found: {file_path}")
        
        df = pd.read_csv(
            file_path,
            sep='::',
            names=['item_id', 'title', 'genres'],
            header=None,
            encoding='latin-1',
            engine='python'
        )
        return df
    
    def load_tags(self) -> pd.DataFrame:
        """Load tags from tags.dat file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "tags.dat"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Tags file not found: {file_path}")
        
        df = pd.read_csv(
            file_path,
            sep='::',
            names=['user_id', 'item_id', 'tag', 'timestamp'],
            header=None,
            engine='python'
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        return df
    
    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all available data for the 10m dataset."""
        return {
            'ratings': self.load_ratings(),
            'movies': self.load_movies(),
            'tags': self.load_tags(),
        }


class MovieLens20mProcessor(MovieLensProcessor):
    """Processor for MovieLens 20m dataset."""
    
    def get_extract_dir(self) -> Path:
        return self.data_base_path / "ml-20m"
    
    def load_ratings(self) -> pd.DataFrame:
        """Load ratings from ratings.csv file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "ratings.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {file_path}")
        
        df = pd.read_csv(file_path, engine='python')
        df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        return df
    
    def load_movies(self) -> pd.DataFrame:
        """Load movie information from movies.csv file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "movies.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Movies file not found: {file_path}")
        
        df = pd.read_csv(file_path, engine='python')
        df = df.rename(columns={'movieId': 'item_id'})
        return df
    
    def load_tags(self) -> pd.DataFrame:
        """Load tags from tags.csv file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "tags.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Tags file not found: {file_path}")
        
        df = pd.read_csv(file_path, engine='python')
        df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        return df
    
    def load_genome_scores(self) -> pd.DataFrame:
        """Load tag genome scores from genome-scores.csv file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "genome-scores.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Genome scores file not found: {file_path}")
        
        df = pd.read_csv(file_path, engine='python')
        df = df.rename(columns={'movieId': 'item_id'})
        return df
    
    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all available data for the 20m dataset."""
        return {
            'ratings': self.load_ratings(),
            'movies': self.load_movies(),
            'tags': self.load_tags(),
            'genome_scores': self.load_genome_scores(),
        }


class MovieLens32mProcessor(MovieLensProcessor):
    """Processor for MovieLens 32m dataset (similar to 20m)."""
    
    def get_extract_dir(self) -> Path:
        return self.data_base_path / "ml-32m"
    
    def load_ratings(self) -> pd.DataFrame:
        """Load ratings from ratings.csv file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "ratings.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {file_path}")
        
        df = pd.read_csv(file_path, engine='python')
        df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        return df
    
    def load_movies(self) -> pd.DataFrame:
        """Load movie information from movies.csv file."""
        extract_dir = self.get_extract_dir()
        file_path = extract_dir / "movies.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Movies file not found: {file_path}")
        
        df = pd.read_csv(file_path, engine='python')
        df = df.rename(columns={'movieId': 'item_id'})
        return df
    
    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all available data for the 32m dataset."""
        return {
            'ratings': self.load_ratings(),
            'movies': self.load_movies(),
        }


class MovieLens(DataLoader):
    """
    Loader for the MovieLens Dataset.
    
    Downloads and loads MovieLens datasets of various sizes.
    Data is saved to the data folder parallel to the src folder in the root directory.
    """
    
    # Get project root (two levels up from this file: src/recsys/data.py -> root)
    _PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_BASE_PATH = _PROJECT_ROOT / "data" / "movielens"
    VALID_SIZES = ["100k", "1m", "10m", "20m", "32m"]
    
    # Map dataset sizes to their processors
    _PROCESSOR_MAP: dict[str, type[MovieLensProcessor]] = {
        "100k": MovieLens100kProcessor,
        "1m": MovieLens1mProcessor,
        "10m": MovieLens10mProcessor,
        "20m": MovieLens20mProcessor,
        "32m": MovieLens32mProcessor,
    }
    
    def __init__(self):
        self.downloader = MovieLensDownloader()
    
    def _get_processor(self, size: str) -> MovieLensProcessor:
        """Get the appropriate processor for the dataset size."""
        size_lc = size.lower()
        if size_lc not in self._PROCESSOR_MAP:
            raise ValueError(f"Unsupported dataset size: {size}")
        
        processor_class = self._PROCESSOR_MAP[size_lc]
        return processor_class(self.DATA_BASE_PATH)
    
    def download_and_extract(self, size: str = '100k'):
        """
        Downloads and extracts the MovieLens dataset of the given size if not already present.
        
        Args:
            size: Dataset size ('100k', '1m', '10m', '20m', '32m')
        """
        self.downloader.download_and_extract(size)
    
    def load_data(self, size: str = '100k') -> dict[str, pd.DataFrame]:
        """
        Loads all MovieLens data into a dictionary of DataFrames.
        Automatically downloads and extracts if necessary.
        
        Args:
            size: Dataset size ('100k', '1m', '10m', '20m', '32m')
            
        Returns:
            Dictionary containing all available data for the dataset:
            - '100k': {'ratings', 'movies', 'users'}
            - '1m': {'ratings', 'movies', 'users'}
            - '10m': {'ratings', 'movies', 'tags'}
            - '20m': {'ratings', 'movies', 'tags', 'genome_scores'}
            - '32m': {'ratings', 'movies'}
        """
        size_lc = size.lower()
        processor = self._get_processor(size_lc)
        
        # Download if necessary
        extract_dir = processor.get_extract_dir()
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            self.download_and_extract(size_lc)
        
        return processor.load_all()