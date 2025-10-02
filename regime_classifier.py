"""Optional machine-learning classifier for market regimes.

Models are loaded if available (Redis key or local pickle). When absent,
this module degrades gracefully so the heuristic regime detector keeps
working unchanged.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

logger = logging.getLogger(__name__)

try:  # Redis is optional
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None  # type: ignore

try:  # scikit-learn is optional at runtime (already used elsewhere in repo)
    from sklearn.base import BaseEstimator
except ImportError:  # pragma: no cover - optional dependency
    BaseEstimator = object  # type: ignore


MODEL_REDIS_KEY = "regime:classifier:model"
MODEL_META_KEY = "regime:classifier:meta"
LOCAL_MODEL_PATH = Path("regime_classifier.pkl")

# Canonical class order used by the repo.
CLASS_LABELS: Sequence[str] = ("ranging", "trending", "volatile")

# Feature ordering must stay in sync with `enhanced_market_regime._compute_regime_features`.
FEATURE_ORDER: Sequence[str] = (
    "trend_slope_15",
    "trend_r2_15",
    "trend_slope_60",
    "trend_r2_60",
    "trend_slope_240",
    "trend_r2_240",
    "ema_alignment",
    "atr_to_price",
    "atr_volatility_ratio",
    "bb_width_percentile",
    "hurst_exponent",
    "auto_corr_1",
    "auto_corr_2",
    "volume_zscore",
    "return_volatility",
    "price_chop_ratio",
)


@dataclass
class _ClassifierPayload:
    model: BaseEstimator
    feature_order: Sequence[str]
    classes: Sequence[str]


class RegimeClassifier:
    """Loads and serves regime predictions if a model is available."""

    def __init__(self) -> None:
        self.payload: Optional[_ClassifierPayload] = None
        self._load_model()

    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self.payload is not None

    # ------------------------------------------------------------------
    def predict_probabilities(self, features: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Return class probabilities or ``None`` when no model is loaded."""
        if not self.payload:
            return None

        order = list(self.payload.feature_order or FEATURE_ORDER)
        vector = [float(features.get(name, 0.0)) for name in order]

        try:
            proba = self.payload.model.predict_proba([vector])[0]
        except Exception as exc:  # pragma: no cover - guard against mismatched models
            logger.warning(f"Regime classifier predict_proba failed: {exc}")
            return None

        classes = list(getattr(self.payload.model, "classes_", self.payload.classes))
        probabilities = {label: float(prob) for label, prob in zip(classes, proba)}

        # Ensure all canonical labels exist
        for label in CLASS_LABELS:
            probabilities.setdefault(label, 0.0)

        return probabilities

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Attempt to hydrate classifier from Redis first, then local disk."""
        # Redis is optional but preferred so Railways deployments stay centralised.
        if not self.payload:
            redis_payload = self._load_from_redis()
            if redis_payload:
                self.payload = redis_payload
                logger.info("Regime classifier loaded from Redis cache")
                return

        # Fallback: local pickle checked into persistent volume or mounted storage.
        if not self.payload:
            local_payload = self._load_from_disk()
            if local_payload:
                self.payload = local_payload
                logger.info("Regime classifier loaded from local pickle")

        if not self.payload:
            logger.info("Regime classifier unavailable – using heuristic regime detection only")

    # ------------------------------------------------------------------
    def _load_from_redis(self) -> Optional[_ClassifierPayload]:
        redis_url = os.getenv("REDIS_URL")
        if not redis_url or redis is None:
            return None

        try:
            client = redis.from_url(redis_url, decode_responses=True)
            raw_model = client.get(MODEL_REDIS_KEY)
            meta_json = client.get(MODEL_META_KEY)
            if not raw_model:
                return None

            model_bytes = base64.b64decode(raw_model)
            model = pickle.loads(model_bytes)

            if meta_json:
                meta = json.loads(meta_json)
                feature_order = meta.get("feature_order", FEATURE_ORDER)
                classes = meta.get("classes", CLASS_LABELS)
            else:
                feature_order, classes = FEATURE_ORDER, CLASS_LABELS

            return _ClassifierPayload(model=model, feature_order=feature_order, classes=classes)
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.warning(f"Failed to load regime classifier from Redis: {exc}")
            return None

    # ------------------------------------------------------------------
    def _load_from_disk(self) -> Optional[_ClassifierPayload]:
        path = Path(os.getenv("REGIME_CLASSIFIER_PATH", LOCAL_MODEL_PATH))
        if not path.exists():
            return None

        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.warning(f"Failed to load regime classifier from {path}: {exc}")
            return None

        # Accept pipeline dumps or raw estimators packaged with metadata
        if isinstance(payload, dict) and "model" in payload:
            model = payload["model"]
            feature_order = payload.get("feature_order", FEATURE_ORDER)
            classes = payload.get("classes", CLASS_LABELS)
        else:
            model = payload
            feature_order = FEATURE_ORDER
            classes = getattr(model, "classes_", CLASS_LABELS)

        return _ClassifierPayload(model=model, feature_order=feature_order, classes=classes)


_classifier_instance: Optional[RegimeClassifier] = None


def get_regime_classifier() -> Optional[RegimeClassifier]:
    """Return singleton-like classifier instance (or ``None`` when unavailable)."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = RegimeClassifier()
    return _classifier_instance if _classifier_instance.is_ready else None
