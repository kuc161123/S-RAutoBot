from typing import List, Tuple, Dict
import json
import logging

from ml_qscore_adapter_base import QScoreAdapterBase

logger = logging.getLogger(__name__)


class ScalpQAdapter(QScoreAdapterBase):
    """Qscore adapter for Scalp using scalp_phantom_tracker data."""

    def __init__(self):
        super().__init__('qcal:scalp')

    def _load_training_records(self) -> List[Tuple[float, int, float, Dict]]:
        out: List[Tuple[float, int, float, Dict]] = []
        try:
            if not self.redis:
                return out
            raw = self.redis.get('scalp_phantom:completed')
            if not raw:
                return out
            arr = json.loads(raw)
            for rec in arr:
                try:
                    feats = rec.get('features') or {}
                    q = feats.get('qscore')
                    if q is None:
                        continue
                    y = 1 if str(rec.get('outcome','')).lower() == 'win' else 0
                    pnl = float(rec.get('pnl_percent', 0.0) or 0.0)
                    ctx = {
                        'session': feats.get('session', 'global'),
                        'volatility_regime': feats.get('volatility_regime', 'global')
                    }
                    out.append((float(q), y, pnl, ctx))
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"ScalpQAdapter load error: {e}")
        return out


_INSTANCE: ScalpQAdapter | None = None


def get_scalp_qadapter() -> ScalpQAdapter:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ScalpQAdapter()
    return _INSTANCE

