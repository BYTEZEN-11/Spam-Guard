import os
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.config.config import Config
from src.utils.state import TrainingState

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self):
        self.config = Config()

    def load_data(self, state: TrainingState) -> TrainingState:
        try:
            logger.info("Loading data")
            path = Path(self.config.training_data_path)
            if not path.is_file():
                raise FileNotFoundError(f"Training data not found: {path}")

            max_bytes = 200 * 1024 * 1024
            size = path.stat().st_size
            if size > max_bytes:
                raise ValueError(
                    f"Training data file is too large "
                    f"({size / (1024 * 1024):.1f} MB > {max_bytes // (1024 * 1024)} MB cap)"
                )

            state.training_data = pd.read_csv(path)
            logger.info(
                f"Data loaded successfully: shape={state.training_data.shape}, "
                f"size={size} bytes"
            )
            return state
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise e
    