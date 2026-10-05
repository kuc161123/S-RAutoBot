"""Causal observation equivalence and fail-closed offline tape checkpoints."""

from collections import Counter
from concurrent.futures import Future
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import json

import pytest

from apex_bot import leverage_research_tapes as tapes
from apex_bot.engine import DAY, H4
from apex_bot.replay import HOUR, _AnalysisCache, _source
from apex_bot.spot_comparison import _observations

from .test_replay import watch

START = 120 * DAY
END = START + 2 * DAY
CUTOFF = START + 8 * HOUR


@pytest.fixture
def archive(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        dict(
            start=pd.date_range("1970-01-01", periods=122 * 24, freq="h"),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1.0,
        )
    )
    path = tmp_path / "TESTUSDT.parquet"
    frame.to_parquet(path, index=False)
    return path, frame


def _watch(identity, now, side="Buy", reason="AWAIT_ZONE"):
    original = watch(identity, now, side)
    return replace(
        original,
        created_at=START - DAY,
        expires_at=START + 29 * DAY,
        reason=reason,
        evidence=dict(
            original.evidence,
            as_of=now,
            daily_closed_at=now // DAY * DAY,
            execution_closed_at=now // H4 * H4,
            monitored_started_at=START - DAY,
        ),
    )


def _inject(monkeypatch, sides=("Buy",), conflict=False):
    def analyzer(symbol, daily, execution, now, **kwargs):
        assert symbol == "TESTUSDT"
        assert kwargs["lifetime_seconds"] == 30 * DAY
        assert all(bar.open_time / 1000 + DAY <= now for bar in daily)
        assert all(bar.open_time / 1000 + H4 <= now for bar in execution)
        reason = "CONFLICTING_DIRECTIONS" if conflict else "AWAIT_ZONE"
        return [_watch(side, now, side, reason) for side in sides]

    monkeypatch.setattr(tapes, "analyze_monitored_zones", analyzer)
    return analyzer


def _build(archive, tmp_path, **changes):
    options = dict(
        source_path=archive[0],
        symbol="TESTUSDT",
        start=START,
        end=END,
        entry_cutoff=CUTOFF,
        output_path=tmp_path / "tape.json.gz",
    )
    options.update(changes)
    return tapes.build_tape(**options)


def _load(archive, tmp_path, **changes):
    options = dict(
        filepath=tmp_path / "tape.json.gz",
        source_path=archive[0],
        start=START,
        end=END,
        entry_cutoff=CUTOFF,
    )
    options.update(changes)
    return tapes.load_tape(**options)


@pytest.mark.parametrize("sides", [("Buy",), ("Sell",), ("Buy", "Sell")])
def test_matches_direct_uncached_hourly_observations_and_keeps_retries(
    archive,
    tmp_path,
    monkeypatch,
    sides,
):
    analyzer = _inject(monkeypatch, sides)
    result = _build(archive, tmp_path)
    daily, execution, hourly, _, _, _ = _source(
        archive[0],
        2,
        datetime.fromtimestamp(END, timezone.utc).isoformat(),
    )
    cache = _AnalysisCache(
        "TESTUSDT",
        daily,
        execution,
        False,
        analyzer,
        {"lifetime_seconds": 30 * DAY},
    )
    prices = {b.open_time // 1000 + HOUR: b for b in hourly}
    terminals, expected, reasons = {}, {}, Counter()
    for now in range(START, CUTOFF, HOUR):
        observed = _observations(cache, terminals, prices[now], now)
        reasons.update(op.reason for op in observed)
        ready = [op.to_dict() for op in observed if op.state == "READY"]
        if ready:
            expected[str(now)] = ready
    assert result["ready_by_time"] == expected
    assert result["reason_counts"] == dict(reasons)
    assert result["ready_ids"] == sorted(sides)
    assert result["scan_count"] == 8
    assert result["ready_observation_count"] == 8 * len(sides)
    assert set(result["ready_by_time"]) == {str(t) for t in range(START, CUTOFF, HOUR)}
    assert result["source"]["initial_daily_bars"] == 120
    assert _load(archive, tmp_path) == result


def test_invalid_opposing_watch_is_not_revived_and_all_reasons_are_counted(
    archive,
    tmp_path,
    monkeypatch,
):
    _inject(monkeypatch, ("Buy", "Sell"), conflict=True)
    archive[1].loc[120 * 24, "high"] = 107.0
    archive[1].to_parquet(archive[0], index=False)
    result = _build(archive, tmp_path)
    assert str(START) not in result["ready_by_time"]
    assert result["reason_counts"]["CONFLICTING_DIRECTIONS"] == 2
    assert result["ready_ids"] == ["Buy"]
    assert len(result["ready_by_time"]) == 7
    assert sum(result["reason_counts"].values()) == 16
    assert all(
        [op["side"] for op in row] == ["Buy"]
        for row in result["ready_by_time"].values()
    )


def test_future_prices_cannot_change_earlier_tape(archive, tmp_path, monkeypatch):
    _inject(monkeypatch)
    before = _build(archive, tmp_path)
    archive[1].loc[120 * 24 + 8 :, ["open", "high", "low", "close"]] *= 2
    archive[1].to_parquet(archive[0], index=False)
    after = _build(archive, tmp_path, output_path=tmp_path / "changed.json.gz")
    assert before["source"]["sha256"] != after["source"]["sha256"]
    assert before["ready_by_time"] == after["ready_by_time"]
    assert before["reason_counts"] == after["reason_counts"]


@pytest.mark.parametrize("damage", ["window_gap", "warmup", "start_quote"])
def test_rejects_incomplete_or_insufficient_history(
    archive,
    tmp_path,
    monkeypatch,
    damage,
):
    if damage == "window_gap":
        archive[1].drop(index=121 * 24).to_parquet(archive[0], index=False)
        match = "incomplete or unaligned"
    elif damage == "warmup":
        archive[1].iloc[24:].to_parquet(archive[0], index=False)
        match = "120 closed daily"
    else:
        original = tapes._source

        def without_start_quote(*args):
            data = list(original(*args))
            data[2] = [b for b in data[2] if b.open_time // 1000 + HOUR != START]
            return data

        monkeypatch.setattr(tapes, "_source", without_start_quote)
        match = "missing hourly quotes"
    with pytest.raises(ValueError, match=match):
        _build(archive, tmp_path)
    assert not (tmp_path / "tape.json.gz").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"start": START + 1},
        {"end": float("nan")},
        {"entry_cutoff": END + HOUR},
        {"entry_cutoff": START},
        {"start": True},
        {"symbol": "../../unsafe"},
    ],
)
def test_rejects_ambiguous_parameters(archive, tmp_path, change):
    with pytest.raises(ValueError):
        _build(archive, tmp_path, **change)


@pytest.mark.parametrize("damage", ["source", "code", "parameters", "payload", "gzip"])
def test_checkpoint_reuse_fails_closed(archive, tmp_path, monkeypatch, damage):
    _inject(monkeypatch)
    result = _build(archive, tmp_path)
    changes = {}
    if damage == "source":
        archive[1].loc[0, "volume"] = 2
        archive[1].to_parquet(archive[0], index=False)
    elif damage == "code":
        monkeypatch.setattr(tapes, "_code_hashes", lambda: {"changed": "hash"})
    elif damage == "parameters":
        changes["entry_cutoff"] = CUTOFF + HOUR
    elif damage == "payload":
        result["ready_by_time"] = {}
        with gzip.open(tmp_path / "tape.json.gz", "wt") as handle:
            json.dump(result, handle)
    else:
        (tmp_path / "tape.json.gz").write_bytes(b"incomplete")
    with pytest.raises(ValueError):
        _load(archive, tmp_path, **changes)


def test_changed_source_during_build_does_not_publish(archive, tmp_path, monkeypatch):
    _inject(monkeypatch)
    original = tapes._observations

    def mutate_once(*args):
        if args[-1] == START:
            archive[1].loc[0, "volume"] = 2
            archive[1].to_parquet(archive[0], index=False)
        return original(*args)

    monkeypatch.setattr(tapes, "_observations", mutate_once)
    with pytest.raises(ValueError, match="changed while building"):
        _build(archive, tmp_path)
    assert not (tmp_path / "tape.json.gz").exists()


def test_existing_checkpoint_is_not_overwritten(archive, tmp_path, monkeypatch):
    _inject(monkeypatch)
    _build(archive, tmp_path)
    original = (tmp_path / "tape.json.gz").read_bytes()
    with pytest.raises(FileExistsError):
        _build(archive, tmp_path)
    assert (tmp_path / "tape.json.gz").read_bytes() == original


def test_cli_build_resume_and_reject_mismatched_manifest(
    archive,
    tmp_path,
    monkeypatch,
    capsys,
):
    _inject(monkeypatch)

    class LocalPool:
        def __init__(self, max_workers):
            assert max_workers == 4

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, job):
            future = Future()
            future.set_result(function(job))
            return future

    monkeypatch.setattr(tapes, "ProcessPoolExecutor", LocalPool)
    manifest = tmp_path / "input.json"
    payload = dict(
        symbols=["TESTUSDT"],
        data_paths={"TESTUSDT": archive[0].name},
        start=START,
        end=END,
        entry_cutoff=CUTOFF,
    )
    manifest.write_text(json.dumps(payload))
    output = tmp_path / "tapes"
    args = ["--manifest", str(manifest), "--outputDIR", str(output)]
    initial = tapes.main(args)
    assert initial["TESTUSDT"]["resumed"] is False
    resumed = tapes.main(args)
    assert resumed["TESTUSDT"]["resumed"] is True
    assert resumed["TESTUSDT"]["ready_observations"] == 8
    payload["entry_cutoff"] += HOUR
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Output manifest"):
        tapes.main(args)
    assert '"complete":true' in capsys.readouterr().out
