import mailbox
import pickle
import time
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

from src.utils.state import PredictionState
from src.utils.logger import get_logger
from src.config.config import Config
from src.utils.email_utils import extract_body, all_recipients, clean_text

logger = get_logger(__name__)

MAX_EMAIL_LENGTH = 200_000
MAX_MBOX_EMAILS = 50_000


class PredictionPipeline:
    def __init__(self, load_models: bool = True):
        self.config = Config()
        self.mailbox = None
        self.feature_transformer = None
        self.model = None

        if load_models:
            self._load_models()

    def _load_models(self) -> None:
        """Load pickled vectorizer and model from trusted, configured paths.

        These paths are taken from ``Config`` (hard-coded by the operator).
        Do NOT load pickles from arbitrary user-provided paths — ``pickle``
        is unsafe with untrusted input.
        """
        logger.info("Loading models...")
        model_path = Path(self.config.model_path)
        feature_path = Path(self.config.feature_path)

        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not feature_path.is_file():
            raise FileNotFoundError(f"Vectorizer file not found: {feature_path}")

        max_bytes = 500 * 1024 * 1024
        for p in (feature_path, model_path):
            size = p.stat().st_size
            if size > max_bytes:
                raise ValueError(
                    f"Refusing to load {p}: size {size} bytes exceeds cap "
                    f"{max_bytes} bytes"
                )

        try:
            with open(feature_path, "rb") as f:
                self.feature_transformer = pickle.load(f)
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to unpickle model artifacts ({feature_path.name}, "
                f"{model_path.name}). The artifacts were likely produced by a "
                f"different scikit-learn version. Retrain the model with "
                f"`python -m src.pipeline.training_pipeline`. Underlying "
                f"error: {exc}"
            ) from exc
        logger.info("Models loaded successfully")

    @staticmethod
    def _label_from_prediction(raw: int) -> str:
        try:
            return "Spam" if int(raw) == 0 else "Ham"
        except (TypeError, ValueError):
            return "Ham"

    def predict_single_email(self, email_body: str) -> Dict:
        if self.model is None or self.feature_transformer is None:
            self._load_models()

        if not isinstance(email_body, str) or not email_body.strip():
            raise ValueError("email_body must be a non-empty string")

        if len(email_body) > MAX_EMAIL_LENGTH:
            raise ValueError(
                f"email_body exceeds maximum length of {MAX_EMAIL_LENGTH} characters"
            )

        cleaned_body = clean_text(email_body)
        features = self.feature_transformer.transform([cleaned_body])
        prediction = self.model.predict(features)
        prediction_label = self._label_from_prediction(prediction[0])

        confidence: Optional[float] = None
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(features)
                confidence = float(max(proba[0])) * 100
        except Exception as exc:
            logger.debug(f"predict_proba unavailable: {exc}")

        return {
            "prediction": prediction_label,
            "confidence": confidence,
            "raw_prediction": int(prediction[0]),
        }

    def load_mailbox(self, mailbox_path: str) -> None:
        """Load an MBOX file from a trusted local path."""
        path = Path(mailbox_path)
        if not path.is_file():
            raise FileNotFoundError(f"Mailbox file not found: {mailbox_path}")
        logger.info(f"Loading mailbox from {mailbox_path}")
        self.mailbox = mailbox.mbox(str(path))
        logger.info(f"Loaded mailbox from {mailbox_path}")

    def process_mailbox(self, mailbox_path: Optional[str] = None) -> List[Dict]:
        if mailbox_path:
            self.load_mailbox(mailbox_path)

        if self.mailbox is None:
            raise ValueError("No mailbox loaded. Call load_mailbox() first.")

        logger.info("Processing mailbox")
        data: List[Dict] = []
        count = 0

        try:
            for message in self.mailbox:
                if count >= MAX_MBOX_EMAILS:
                    logger.warning(
                        f"Reached MAX_MBOX_EMAILS={MAX_MBOX_EMAILS}; skipping remainder"
                    )
                    break

                labels = (message.get("X-Gmail-Labels") or "").lower()
                category = (
                    "Spam" if "spam" in labels else
                    "Promotions" if "category_promotions" in labels else
                    "Social" if "category_social" in labels else
                    "Updates" if "category_updates" in labels else
                    "Inbox"
                )
                time_str = message.get("Date", "")
                recipients = clean_text(all_recipients(message))
                subject = clean_text(message.get("Subject", ""))
                body = clean_text(extract_body(message))
                direction = "Sent" if "Sent" in (message.get("X-Gmail-Labels") or "") else "Received"

                data.append({
                    "Time": time_str,
                    "Recipients": recipients,
                    "Subject": subject,
                    "Body": body,
                    "Category": category,
                    "Direction": direction,
                })
                count += 1
        finally:
            try:
                self.mailbox.close()
            except Exception as exc:
                logger.debug(f"mailbox.close() failed: {exc}")

        logger.info(f"Processed {len(data)} emails from mailbox")

        return data

    def run_prediction(self, mail_data: List[Dict]) -> List[Dict]:
        if self.model is None or self.feature_transformer is None:
            self._load_models()

        if not mail_data:
            return mail_data

        start_time = time.time()
        logger.info(f"Running predictions on {len(mail_data)} emails")

        has_proba = hasattr(self.model, "predict_proba")

        for mail in mail_data:
            body_text = mail.get("Body", "") or ""
            try:
                features = self.feature_transformer.transform([body_text])
                prediction = self.model.predict(features)
                mail["Prediction"] = self._label_from_prediction(prediction[0])
                mail["Raw_Prediction"] = int(prediction[0])

                if has_proba:
                    proba = self.model.predict_proba(features)
                    mail["Confidence"] = float(max(proba[0])) * 100
            except Exception as exc:
                logger.error(f"Failed to predict email: {exc}")
                mail["Prediction"] = "Error"
                mail["Confidence"] = None

        end_time = time.time()
        logger.info(f"Prediction completed in {end_time - start_time:.2f} seconds")

        return mail_data

    def predict_mbox_file(
        self,
        mailbox_path: str,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        mail_data = self.process_mailbox(mailbox_path)
        mail_data = self.run_prediction(mail_data)
        df = pd.DataFrame(mail_data)
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        return df


def run_legacy_pipeline(state: PredictionState) -> None:
    pipeline = PredictionPipeline(load_models=False)
    pipeline.load_mailbox(state.mailbox_path)
    mail_data = pipeline.process_mailbox()
    state.mail_data = mail_data
    state.mail_data = pipeline.run_prediction(state.mail_data)
    df = pd.DataFrame(state.mail_data)
    df.to_csv("data/predictions.csv", index=False)