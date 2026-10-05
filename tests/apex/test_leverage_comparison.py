"""Shared-book leverage integration using real risk, fills and margin accounting."""

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from apex_bot import leverage_comparison as comparison
from apex_bot import leverage_research_accounting as accounting
from apex_bot import leverage_research_tapes as tapes
from apex_bot.engine import DAY, H4
from apex_bot.replay import HOUR

from .test_leverage_research_tapes import START, END, archive, _watch


@pytest.fixture
def book(archive, tmp_path):
    pd = pytest.importorskip("pandas")
    count = 0

    def make(symbols=("BTCUSDT",), *, cutoff=START + DAY, rates=None):
        nonlocal count
        count += 1
        root = tmp_path / f"book-{count}"
        root.mkdir()
        manifest = dict(
            symbols=list(symbols),
            start=START,
            end=END,
            entry_cutoff=cutoff,
            data_paths={},
            funding_paths={},
        )
        frames = {}
        for symbol in symbols:
            path = root / f"{symbol}.parquet"
            frames[symbol] = archive[1].copy()
            frames[symbol].to_parquet(path, index=False)
            funding_path = root / f"{symbol}-funding.parquet"
            times = list(range(START - DAY, END + DAY + 1, 8 * HOUR))
            pd.DataFrame(
                dict(
                    ts_ms=[t * 1000 for t in times],
                    funding_rate=[(rates or {}).get(t, 0) for t in times],
                )
            ).to_parquet(funding_path, index=False)
            manifest["data_paths"][symbol] = str(path)
            manifest["funding_paths"][symbol] = str(funding_path)
        return SimpleNamespace(manifest=manifest, frames=frames, tapes=root / "tapes")

    return make


def _op(symbol, now, side="Buy", identity=None, wide=False):
    original = _watch(identity or symbol, now, side)
    return replace(
        original,
        symbol=symbol,
        bucket="majors" if symbol in ("BTCUSDT", "ETHUSDT") else "alts",
        stop=(94 if wide else 96.5) if side == "Buy" else (106 if wide else 103.5),
        invalidation=98.5 if side == "Buy" else 101.5,
    )


def _prepare(book, monkeypatch, factory=None):
    factory = factory or (lambda symbol, now: [_op(symbol, now)])

    def analyzer(symbol, daily, execution, now, **kwargs):
        assert all(b.open_time / 1000 + DAY <= now for b in daily)
        assert all(b.open_time / 1000 + H4 <= now for b in execution)
        return factory(symbol, now)

    monkeypatch.setattr(tapes, "analyze_monitored_zones", analyzer)
    for symbol in book.manifest["symbols"]:
        path = book.manifest["data_paths"][symbol]
        book.frames[symbol].to_parquet(path, index=False)
        tapes.build_tape(
            path,
            symbol,
            START,
            END,
            book.manifest["entry_cutoff"],
            book.tapes / f"{symbol}.json.gz",
        )


def _run(book, leverage=5, mmr=0.005):
    return comparison.run_scenario(book.manifest, book.tapes, leverage, mmr)


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_fixed_risk_five_and_ten_have_same_quantity_and_cashflows(
    book,
    monkeypatch,
    side,
):
    data = book()
    # Resolve both targets after the entry bar, with no stop touched.
    data.frames["BTCUSDT"].loc[120 * 24 + 1, "high" if side == "Buy" else "low"] = (
        131 if side == "Buy" else 69
    )
    _prepare(data, monkeypatch, lambda s, n: [_op(s, n, side)])
    five, ten = _run(data, 5), _run(data, 10)
    assert five["accepted"] == ten["accepted"] == 1
    assert five["closed"] == ten["closed"] == 1
    for key in ("net_equity_change", "total_fees", "funding_estimate", "wins"):
        assert five[key] == pytest.approx(ten[key])
    for key in ("qty", "risk_cash", "entry", "stop"):
        assert five["accepted_evidence"][0]["sizing"][key] == (
            ten["accepted_evidence"][0]["sizing"][key]
        )
    assert five["max_reserved_margin_usdt"] > ten["max_reserved_margin_usdt"]
    assert five["per_side"][side]["closed"] == 1
    assert five["source_unchanged"] and five["funding_source_unchanged"]


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_ten_x_guard_rejects_wide_stop_that_five_x_accepts(book, monkeypatch, side):
    data = book()
    _prepare(data, monkeypatch, lambda s, n: [_op(s, n, side, wide=True)])
    five, ten = _run(data, 5), _run(data, 10)
    assert five["accepted"] == 1
    assert ten["accepted"] == 0
    assert (
        ten["unique_risk_rejections"]["ESTIMATED_LIQUIDATION_BUFFER_INSUFFICIENT"] == 1
    )
    assert ten["risk_rejections"]["ESTIMATED_LIQUIDATION_BUFFER_INSUFFICIENT"] == 24
    assert ten["net_equity_change"] == 0


def test_shared_book_pending_reservations_and_bucket_caps(book, monkeypatch):
    data = book(("SOLUSDT", "LINKUSDT", "ETHUSDT", "BTCUSDT", "BNBUSDT"))
    _prepare(
        data, monkeypatch, lambda s, n: [_op(s, n, "Sell" if s == "ETHUSDT" else "Buy")]
    )
    result = _run(data)
    evidence = result["accepted_evidence"]
    assert [x["opportunity"]["symbol"] for x in evidence] == [
        "BNBUSDT",
        "BTCUSDT",
        "ETHUSDT",
        "LINKUSDT",
    ]
    reserved = 0
    for index, row in enumerate(evidence):
        before, sizing, op = row["account_before"], row["sizing"], row["opportunity"]
        assert before["equity"] == 10000
        assert before["pending_count"] == index
        assert before["reserved_pending"] == pytest.approx(reserved)
        assert before["available_cash"] == pytest.approx(10000 - reserved)
        reserved += accounting.pending_requirement(
            sizing["qty"],
            sizing["entry"],
            5,
            upper_fill_price=(
                op["evidence"]["zone_high"] if op["side"] == "Sell" else None
            ),
        )
    assert result["maximum_positions_and_reservations"] == 4
    assert result["unique_risk_rejections"]["BUCKET_POSITION_CAP"] == 1
    assert result["minimum_available_cash"] > 0
    assert result["initial_equity"] == 10000


def test_short_precheck_uses_same_upper_fill_bound_as_pending_account(
    book, monkeypatch
):
    data = book()
    _prepare(data, monkeypatch, lambda s, n: [_op(s, n, "Sell")])
    calls = []
    original = accounting.pending_requirement

    def observe(*args, **kwargs):
        calls.append(kwargs.get("upper_fill_price"))
        return original(*args, **kwargs)

    monkeypatch.setattr(accounting, "pending_requirement", observe)
    result = _run(data)
    assert result["accepted"] == 1
    assert calls and calls[0] == 101


def test_cutoff_stops_new_decisions_but_allows_pre_cutoff_entry_and_management(
    book, monkeypatch
):
    data = book(cutoff=START + HOUR)
    data.frames["BTCUSDT"].loc[120 * 24 + 1, "high"] = 131
    _prepare(data, monkeypatch)
    result = _run(data)
    assert result["accepted"] == result["filled"] == result["closed"] == 1
    trade = result["trades"][0]
    assert trade["opened_at"] == START
    assert trade["closed_at"] > START + HOUR
    assert all(row["time"] < START + HOUR for row in result["accepted_evidence"])


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_funding_is_settled_chronologically_and_never_used_early(
    book, monkeypatch, side
):
    rates = {START: 0.0001, START + 8 * HOUR: 0.0002, START + 16 * HOUR: -0.0001}
    data = book(rates=rates)
    _prepare(
        data,
        monkeypatch,
        lambda s, n: [
            _op(s, n, side, "a-first"),
            _op(s, n, side, "z-retry"),
        ],
    )
    assessments = []
    real_model = accounting.LeverageModel

    def model(*args, **kwargs):
        value = real_model(*args, **kwargs)
        assess = value.assess

        def recorded(op, instrument, equity, **inputs):
            assessments.append((inputs["now"], inputs["funding_rate_8h"], equity))
            return assess(op, instrument, equity, **inputs)

        value.assess = recorded
        return value

    monkeypatch.setattr(accounting, "LeverageModel", model)
    result = _run(data)
    assert result["accepted"] == 1
    for now, rate, _ in assessments:
        assert rate == rates[max(t for t in rates if t <= now)]
    trade = result["trades"][0]
    events = trade["funding_events"]
    assert all(e["timestamp_ms"] > START * 1000 for e in events)
    sign = 1 if side == "Buy" else -1
    assert result["funding_estimate"] == pytest.approx(
        -sign * trade["qty"] * trade["entry"] * (0.0002 - 0.0001)
    )
    assert result["final_account"]["wallet_balance"] == pytest.approx(
        10000 - result["total_fees"] + result["funding_estimate"]
    )
    before = next(eq for t, _, eq in assessments if t == START + 7 * HOUR)
    after = next(eq for t, _, eq in assessments if t == START + 8 * HOUR)
    assert after - before == pytest.approx(
        -sign * trade["qty"] * trade["entry"] * 0.0002
    )


def test_funding_eight_hour_normalization_and_six_hour_verified_transition(book):
    import pandas as pd

    data = book(rates={START: 0.0001})
    path = data.manifest["funding_paths"]["BTCUSDT"]
    times, rates, normalized, info = comparison.load_funding(path, START, END)
    assert rates == normalized
    assert set(info["interval_counts_hours"]) == {8}
    frame = pd.read_parquet(path)
    # A documented 8h -> 6h -> 2h -> 8h schedule transition.
    at = (START + 8 * HOUR) * 1000
    frame.loc[frame.ts_ms == at, "ts_ms"] = at - 2 * HOUR * 1000
    frame.loc[frame.ts_ms == at - 2 * HOUR * 1000, "funding_rate"] = 0.0003
    extra = pd.DataFrame(dict(ts_ms=[at], funding_rate=[0.0002]))
    pd.concat([frame, extra], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Unsupported interval"):
        comparison.load_funding(path, START, END)
    times, _, normalized, info = comparison.load_funding(
        path,
        START,
        END,
        [at - 2 * HOUR * 1000],
    )
    assert normalized[times.index(at - 2 * HOUR * 1000)] == pytest.approx(0.0004)
    assert normalized[times.index(at)] == pytest.approx(0.0008)
    assert info["crosschecked_transition_timestamps"] == [at - 2 * HOUR * 1000]
    assert info["complete_exchange_schedule_verified"] is False


def test_funding_rejects_missing_sixteen_hour_gap(book):
    import pandas as pd

    data = book()
    path = data.manifest["funding_paths"]["BTCUSDT"]
    frame = pd.read_parquet(path)
    frame[frame.ts_ms != (START + 8 * HOUR) * 1000].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Unsupported interval"):
        comparison.load_funding(path, START, END)


@pytest.fixture
def certified_book(book, monkeypatch):
    data = book()
    _prepare(data, monkeypatch)
    snapshot = comparison.input_snapshot(data.manifest, data.tapes)
    data.manifest["source_sha256"] = {
        symbol: {kind: value["sha256"] for kind, value in files.items()}
        for symbol, files in snapshot.items()
    }
    data.snapshot = snapshot
    data.audit_path = data.tapes.parent / "selection_audit.json"
    data.audit_path.write_text(json.dumps({"selected": ["BTCUSDT"]}))
    data.manifest["selection_audit_sha256"] = comparison._sha(data.audit_path)
    data.manifest_path = data.tapes.parent / "manifest.json"
    data.manifest_path.write_text(json.dumps(data.manifest))
    data.output = data.tapes.parent / "results"
    return data


@pytest.mark.parametrize("field", ["source", "funding_source", "tape_source"])
@pytest.mark.parametrize("damage", ["mismatch", "missing"])
def test_worker_must_match_every_parent_source_and_tape_hash(
    certified_book,
    field,
    damage,
):
    data = certified_book
    result = _run(data)
    assert result["tape_source_unchanged"]
    assert comparison.scenario_matches_inputs(result, data.snapshot)
    altered = deepcopy(result)
    if damage == "mismatch":
        altered[field]["BTCUSDT"]["sha256"] = "0" * 64
    else:
        del altered[field]["BTCUSDT"]
    assert not comparison.scenario_matches_inputs(altered, data.snapshot)
    assert comparison.scenario_matches_inputs(result, data.snapshot)


def _cli_args(data, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "leverage_comparison",
            "--manifest",
            str(data.manifest_path),
            "--tapes",
            str(data.tapes),
            "--output",
            str(data.output),
        ],
    )


@pytest.mark.parametrize("kind", ["hourly", "funding", "tape"])
def test_parent_rejects_preselected_input_tamper_before_dispatch(
    certified_book,
    monkeypatch,
    kind,
):
    data = certified_book
    path = Path(data.snapshot["BTCUSDT"][kind]["path"])
    path.write_bytes(path.read_bytes() + b"tampered after selection")
    with pytest.raises(ValueError, match=f"{kind} changed since universe selection"):
        comparison.input_snapshot(data.manifest, data.tapes)

    def forbidden_pool(*args, **kwargs):
        pytest.fail("Workers must not start with changed preselected inputs")

    monkeypatch.setattr(comparison, "ProcessPoolExecutor", forbidden_pool)
    _cli_args(data, monkeypatch)
    with pytest.raises(ValueError, match="changed since universe selection"):
        comparison.main()
    assert not data.output.exists()


def _worker_pool(monkeypatch, result, *, wrong_field=None, after_workers=None):
    """Use completed worker evidence to isolate the parent's certification gate."""

    class CompletedPool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            if after_workers is not None:
                after_workers()

        def submit(self, function, manifest, tapes_dir, leverage, mmr, **kwargs):
            row = deepcopy(result)
            row.update(leverage=leverage, maintenance_rate_assumption=mmr)
            if wrong_field is not None and leverage == 10 and mmr == 0.01:
                row[wrong_field]["BTCUSDT"]["sha256"] = "1" * 64
            future = Future()
            future.set_result(row)
            return future

    monkeypatch.setattr(comparison, "ProcessPoolExecutor", CompletedPool)


@pytest.mark.parametrize(
    "wrong_field", [None, "source", "funding_source", "tape_source"]
)
def test_parent_completion_requires_worker_hash_agreement(
    certified_book,
    monkeypatch,
    wrong_field,
):
    data = certified_book
    result = _run(data)
    _worker_pool(monkeypatch, result, wrong_field=wrong_field)
    _cli_args(data, monkeypatch)
    if wrong_field is None:
        comparison.main()
    else:
        with pytest.raises(SystemExit) as exc:
            comparison.main()
        assert exc.value.code == 1
    report = json.loads((data.output / "report.json").read_text())
    assert len(report["rows"]) == 6 and not report["errors"]
    assert report["all_inputs_unchanged"]
    assert report["selection_audit_unchanged"]
    assert report["input_snapshot"] == data.snapshot
    assert report["complete"] is (wrong_field is None)
    assert sum(not r["matches_parent_inputs"] for r in report["rows"]) == (
        0 if wrong_field is None else 1
    )


@pytest.mark.parametrize("kind", ["hourly", "funding", "tape", "selection_audit"])
def test_parent_final_rehash_catches_changes_after_all_worker_certifications(
    certified_book,
    monkeypatch,
    kind,
):
    data = certified_book
    result = _run(data)
    path = (
        data.audit_path
        if kind == "selection_audit"
        else Path(data.snapshot["BTCUSDT"][kind]["path"])
    )

    def mutate_after_workers():
        path.write_bytes(path.read_bytes() + b"changed after worker completion")

    _worker_pool(monkeypatch, result, after_workers=mutate_after_workers)
    _cli_args(data, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        comparison.main()
    assert exc.value.code == 1
    report = json.loads((data.output / "report.json").read_text())
    assert len(report["rows"]) == 6 and not report["errors"]
    # Every worker individually certified its inputs. Only the final parent
    # check detects this change, which previously could certify a mixed run.
    assert all(
        all(
            row[k]
            for k in (
                "source_unchanged",
                "funding_source_unchanged",
                "tape_source_unchanged",
                "matches_parent_inputs",
            )
        )
        for row in report["rows"]
    )
    assert not report["complete"]
    assert report["all_inputs_unchanged"] is (kind == "selection_audit")
    assert report["selection_audit_unchanged"] is (kind != "selection_audit")
