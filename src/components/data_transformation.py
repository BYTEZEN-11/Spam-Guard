import numpy as np
from src.utils.logger import get_logger
from src.config.config import Config
from src.utils.state import TrainingState
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

logger = get_logger(__name__)


class DataTransformation:
    def __init__(self):
        self.config = Config()

    def transform_data(self, state: TrainingState) -> TrainingState:
        logger.info("Data transformation started")
        try:
            if state.training_data is None or state.training_data.empty:
                raise ValueError("No training data available on state.")

            required_cols = {"Message", "Category"}
            missing = required_cols - set(state.training_data.columns)
            if missing:
                raise ValueError(f"Training data missing required columns: {sorted(missing)}")

            data = state.training_data.copy()

            data.loc[data["Category"] == "spam", "Category"] = 0
            data.loc[data["Category"] == "ham", "Category"] = 1

            unique_labels = set(data["Category"].unique())
            unexpected = unique_labels - {0, 1}
            if unexpected:
                raise ValueError(
                    f"Unexpected label values found in Category: {unexpected}. "
                    f"Only 'spam' and 'ham' are supported."
                )

            data["Category"] = data["Category"].astype(int)

            before = len(data)
            data = data.dropna(subset=["Message"]).reset_index(drop=True)
            dropped = before - len(data)
            if dropped:
                logger.info(f"Dropped {dropped} rows with empty Message")

            logger.info(f"Label encoding completed. Data shape: {data.shape}")
            logger.info(f"Unique labels: {data['Category'].unique().tolist()}")
            logger.info(f"Label dtype: {data['Category'].dtype}")

            X = data["Message"]
            y = np.array(data["Category"], dtype=int)

            class_counts = np.bincount(y)
            if class_counts.size < 2 or (class_counts >= 2).sum() < 2:
                raise ValueError(
                    "Stratified split needs at least 2 samples in each class. "
                    f"Got counts: {class_counts.tolist()}"
                )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y,
            )

            logger.info(
                f"Train/test split completed. Train size: {len(X_train)}, Test size: {len(X_test)}"
            )

            tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=20_000,
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 2),
            )
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = tfidf_vectorizer.transform(X_test)

            logger.info(
                f"TF-IDF transformation completed. Feature shape: {X_train_tfidf.shape}"
            )

            state.transformed_data = data
            state.X_train = X_train
            state.X_test = X_test
            state.y_train = y_train
            state.y_test = y_test
            state.X_train_tfidf = X_train_tfidf
            state.X_test_tfidf = X_test_tfidf
            state.tfidf_vectorizer = tfidf_vectorizer

            logger.info("Data transformation completed")
            return state
        except Exception as e:
            logger.error(f"Failed to transform data: {str(e)}")
            raise e