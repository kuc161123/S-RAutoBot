"""Public archive authenticity, timestamp units and gap rejection."""

import hashlib
import io
import zipfile

import pandas as pd
import pytest

from apex_bot import spot_research_data as data


def _archive(name="BTCUSDT-1h-2024-01.csv", content=b"sample"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, content)
    blob = stream.getvalue()
    checksum = f"{hashlib.sha256(blob).hexdigest()}  BTCUSDT-1h-2024-01.zip\n".encode()
    return blob, checksum


def test_verifies_checksum_before_parsing():
    blob, checksum = _archive()
    assert (
        data.validate_archive(blob, checksum, "BTCUSDT-1h-2024-01.zip")[0] == b"sample"
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.validate_archive(blob + b"modified", checksum, "BTCUSDT-1h-2024-01.zip")
    with pytest.raises(ValueError, match="filename mismatch"):
        data.validate_archive(blob, checksum, "ETHUSDT-1h-2024-01.zip")


@pytest.mark.parametrize("name", ["../BTCUSDT-1h-2024-01.csv", "unexpected.csv"])
def test_rejects_unexpected_zip_members(name):
    blob, checksum = _archive(name)
    with pytest.raises(ValueError, match="archive content"):
        data.validate_archive(blob, checksum, "BTCUSDT-1h-2024-01.zip")


@pytest.mark.parametrize(
    "market,month,unit",
    [
        ("spot", "2024-01", "ms"),
        ("spot", "2025-01", "us"),
        ("perpetual", "2025-01", "ms"),
    ],
)
def test_candle_timestamp_units(market, month, unit):
    timestamp = pd.Timestamp(month + "-01", tz="UTC")
    number = timestamp.value // (1_000_000 if unit == "ms" else 1000)
    header = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
        if market == "perpetual"
        else ""
    )
    payload = (header + f"{number},100,102,99,101,3,0,0,0,0,0,0\n").encode()
    frame = data.parse_month(payload, market, month)
    assert frame.start.iloc[0] == timestamp
    assert frame.close.iloc[0] == 101
    with pytest.raises(ValueError, match="outside named month"):
        data.parse_month(
            payload, market, "2024-02" if month == "2024-01" else "2025-02"
        )


def test_preserves_actual_funding_timestamp():
    stamp = pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000 + 13
    payload = f"calc_time,funding_interval_hours,last_funding_rate\n{stamp},8,0.0001\n".encode()
    frame = data.parse_month(payload, "funding", "2024-01")
    assert frame.ts_ms.iloc[0] == stamp
    with pytest.raises(ValueError, match="funding interval"):
        data.parse_month(payload.replace(b",8,", b",3,"), "funding", "2024-01")
    varied = data.parse_month(payload.replace(b",8,", b",4,"), "funding", "2024-01")
    assert varied.funding_interval_hours.iloc[0] == 4


@pytest.fixture
def window(monkeypatch):
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = start + pd.Timedelta(days=1)
    monkeypatch.setattr(data, "START", start)
    monkeypatch.setattr(data, "END", end)
    return start, end


def test_hourly_series_must_be_complete_and_valid(window):
    start, end = window
    frame = pd.DataFrame(
        dict(
            start=pd.date_range(start, end, freq="h", inclusive="left"),
            open=100,
            high=102,
            low=99,
            close=101,
            volume=2.0,
        )
    )
    assert len(data.validate_series(frame, "spot")) == 24
    with pytest.raises(ValueError, match="Noncontiguous"):
        data.validate_series(frame.drop(5), "spot")
    with pytest.raises(ValueError, match="Noncontiguous"):
        data.validate_series(pd.concat([frame, frame.iloc[[5]]]), "spot")
    broken = frame.copy()
    broken.loc[5, "low"] = 103
    with pytest.raises(ValueError, match="Invalid OHLCV"):
        data.validate_series(broken, "spot")
    broken = frame.copy()
    broken.loc[5, "volume"] = float("inf")
    with pytest.raises(ValueError, match="Nonfinite"):
        data.validate_series(broken, "spot")


def test_settlement_coverage_preserves_delays_and_rejects_holes(window):
    start, end = window
    stamps = pd.date_range(start, end, freq="8h").asi8 // 1_000_000
    frame = pd.DataFrame(dict(ts_ms=stamps + 13, funding_rate=0.0001))
    result = data.validate_series(frame, "funding")
    assert list(result.ts_ms) == list(stamps + 13)
    with pytest.raises(ValueError, match="Noncontiguous"):
        data.validate_series(frame.drop(1), "funding")
    broken = frame.copy()
    broken.loc[1, "ts_ms"] += 120000
    with pytest.raises(ValueError, match="delayed"):
        data.validate_series(broken, "funding")


def test_rejects_non_public_download_url():
    with pytest.raises(ValueError, match="official public"):
        data._get("https://example.com/arbitrary")
