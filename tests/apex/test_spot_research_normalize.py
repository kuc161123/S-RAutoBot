"""Coverage selection must precede outcomes and never move the entry window."""

import pandas as pd
import pytest

from apex_bot import spot_research_normalize as normalize


def _candles(missing=()):
    index = pd.date_range(
        "2022-01-01", "2024-01-01", freq="h", tz="UTC", inclusive="left"
    )
    removed = pd.to_datetime(list(missing), utc=True)
    return pd.DataFrame({"start": index[~index.isin(removed)]})


def test_uses_one_common_day_after_latest_pre_entry_gap():
    spot = _candles(["2023-03-24T13:00:00Z"])
    perp = _candles(["2023-04-10T10:00:00Z"])
    origin, gaps = normalize.common_origin(
        {("spot", "BTCUSDT"): spot, ("perpetual", "ETHUSDT"): perp}
    )
    assert origin == pd.Timestamp("2023-04-11", tz="UTC")
    assert len(gaps) == 2
    assert gaps[0]["missing_from"] == "2023-03-24T13:00:00+00:00"
    assert len(spot) == len(_candles()) - 1  # Inputs unchanged, no interpolation.


@pytest.mark.parametrize(
    "missing,error",
    [
        ("2023-12-01T13:00:00Z", "fixed evaluation"),
        ("2023-10-01T13:00:00Z", "120 complete"),
    ],
)
def test_cannot_shorten_test_or_ignore_insufficient_warmup(missing, error):
    with pytest.raises(ValueError, match=error):
        normalize.common_origin({("spot", "BTCUSDT"): _candles([missing])})


def test_rejects_duplicate_history():
    frame = _candles()
    frame = pd.concat([frame.iloc[:1], frame], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        normalize.common_origin({("spot", "BTCUSDT"): frame})
