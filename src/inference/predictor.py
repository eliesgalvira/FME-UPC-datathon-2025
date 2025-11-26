"""Fast inference utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data.preprocessor import NestedFeatureParser
from src.features.engineer import FeatureEngineer
from src.models.buyer_classifier import BuyerClassifier
from src.models.ensemble import StackingEnsemble
from src.models.revenue_regressor import ODMNRevenueRegressor
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


class FastPredictor:
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)
        self.parser = NestedFeatureParser()
        self.engineer = FeatureEngineer(self.config)
        self.buyer_model = BuyerClassifier(self.config)
        self.revenue_model = ODMNRevenueRegressor(self.config)
        self.ensemble_model = StackingEnsemble(self.config)
        self._load_models()

    def _load_models(self) -> None:
        self.buyer_model.load("models/buyer_classifier.txt")
        self.revenue_model.load("models/odmn")
        self.ensemble_model.load("models/stacking_ensemble.pkl")

    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        processed = self.parser.process_all(df)
        engineered = self.engineer.engineer(processed, target_col=None, fit=fit)
        drop_cols = ["row_id", "datetime"]
        numeric = engineered[[c for c in engineered.columns if c not in drop_cols]].select_dtypes(include="number")
        return numeric.fillna(0)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        features = self._prepare_features(df, fit=False)
        buyer_proba = self.buyer_model.predict_proba(features)
        revenue = self.revenue_model.predict(features)
        lambda_loss = self.config["models"]["stage2_revenue"]["loss"]
        meta = pd.DataFrame(
            {
                "buyer_proba": buyer_proba,
                "revenue_d1": revenue["d1"],
                "revenue_d7": revenue["d7"],
                "revenue_d14": revenue["d14"],
                "weighted_revenue": lambda_loss["lambda_d1"] * revenue["d1"]
                + lambda_loss["lambda_d7"] * revenue["d7"]
                + lambda_loss["lambda_d14"] * revenue["d14"],
                "buyer_x_revenue": buyer_proba * revenue["d7"],
            }
        )
        predictions = self.ensemble_model.predict(meta)
        return pd.Series(predictions, index=df.index)

    def predict_file(self, path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        preds = self.predict(df)
        return pd.DataFrame({"row_id": df["row_id"], "iap_revenue_d7": preds})
