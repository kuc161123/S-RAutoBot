"""Offline causal opportunity tapes; no risk, fills, funding or runtime changes.

Times are UTC Unix seconds. Each sparse row is an hourly observation, not a
trade: READY IDs deliberately repeat so portfolio consumers can retry them.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from time import perf_counter

from .engine import DAY
from .monitored_zones import analyze_monitored_zones
from .replay import HOUR, _AnalysisCache, _source
from .spot_comparison import _observations

FORMAT_VERSION = 1
MIN_WARMUP_DAYS = 120
# These are the source/analysis/observation implementation dependencies. Risk
# and management are intentionally outside the tape's calculation and identity.
DEPENDENCIES = (
    "leverage_research_tapes.py",
    "replay.py",
    "spot_comparison.py",
    "models.py",
    "engine.py",
    "monitored_zones.py",
    "zone_orders.py",
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _code_hashes():
    root = Path(__file__).resolve().parent
    return {name: _sha(root / name) for name in DEPENDENCIES}


def _parameters(start, end, entry_cutoff):
    times = (start, end, entry_cutoff)
    if any(
        isinstance(t, bool)
        or not isinstance(t, (int, float))
        or not math.isfinite(t)
        or t < 0
        or t % HOUR
        for t in times
    ):
        raise ValueError("Tape times must be nonnegative, hour-aligned Unix seconds")
    start, end, entry_cutoff = map(int, times)
    if not start < entry_cutoff <= end:
        raise ValueError("Require start < entry_cutoff <= end")
    return dict(
        start=start,
        end=end,
        entry_cutoff=entry_cutoff,
        scan_seconds=HOUR,
        minimum_daily_warmup=MIN_WARMUP_DAYS,
        analyzer="analyze_monitored_zones",
        analyzer_kwargs={"lifetime_seconds": 30 * DAY},
        analysis_cache=True,
        sides=["Buy", "Sell"],
        quote_half_spread=0.00005,
        history_policy="All complete bars from fixed contiguous segment origin",
    )


def _symbol(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z0-9]{2,40}", value):
        raise ValueError("Expected an uppercase alphanumeric symbol")
    return value


def _publish(path, payload, compressed=True):
    """Publish a complete file atomically, refusing to replace existing work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tape-", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        opener = gzip.open if compressed else open
        with opener(temporary, "wt", encoding="utf-8") as handle:
            handle.write(_json(payload))
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_tape(source_path, symbol, start, end, entry_cutoff, output_path):
    """Write and return a validated tape; the output must not already exist."""
    began = perf_counter()
    symbol = _symbol(symbol)
    parameters = _parameters(start, end, entry_cutoff)
    start, end, entry_cutoff = (parameters[k] for k in ("start", "end", "entry_cutoff"))
    if Path(output_path).exists():
        raise FileExistsError(output_path)
    source_path = Path(source_path).resolve()
    source_hash, code_hashes = _sha(source_path), _code_hashes()
    daily, execution, hourly, begin, stop, metadata = _source(
        source_path,
        (end - start) / DAY,
        datetime.fromtimestamp(end, timezone.utc).isoformat(),
    )
    if metadata["window_truncated"] or begin != start or stop != end:
        raise ValueError(f"{symbol}: incomplete or unaligned tape window")
    cache = _AnalysisCache(
        symbol,
        daily,
        execution,
        True,
        analyze_monitored_zones,
        {"lifetime_seconds": 30 * DAY},
    )
    warm_daily, warm_execution = cache.histories(start)
    if len(warm_daily) < MIN_WARMUP_DAYS:
        raise ValueError(f"{symbol}: need at least 120 closed daily warmup bars")
    prices = {bar.open_time // 1000 + HOUR: bar for bar in hourly}
    if not all(now in prices for now in range(start, end + 1, HOUR)):
        raise ValueError(f"{symbol}: missing hourly quotes, including quote at start")
    terminals, ready_by_time, reasons, ready_ids = {}, {}, Counter(), set()
    for now in range(start, entry_cutoff, HOUR):
        ready = []
        for op in _observations(cache, terminals, prices[now], now):
            reasons[op.reason] += 1
            if op.state == "READY":
                ready.append(op.to_dict())
                ready_ids.add(op.id)
        if ready:
            ready_by_time[str(now)] = ready
    if _sha(source_path) != source_hash or _code_hashes() != code_hashes:
        raise ValueError("Source or implementation changed while building tape")
    result = dict(
        format_version=FORMAT_VERSION,
        symbol=symbol,
        start=start,
        end=end,
        cutoff=entry_cutoff,
        parameters=parameters,
        source=dict(
            metadata,
            path=str(source_path),
            sha256=source_hash,
            size_bytes=source_path.stat().st_size,
            initial_daily_bars=len(warm_daily),
            initial_execution_bars=len(warm_execution),
        ),
        code_hashes=code_hashes,
        ready_by_time=ready_by_time,
        reason_counts=dict(sorted(reasons.items())),
        ready_ids=sorted(ready_ids),
        scan_count=(entry_cutoff - start) // HOUR,
        ready_observation_count=sum(map(len, ready_by_time.values())),
        elapsed_seconds=perf_counter() - began,
        complete=True,
    )
    result["payload_sha256"] = _digest(result)
    _publish(output_path, result)
    return result


def load_tape(filepath, source_path, start, end, entry_cutoff):
    """Fail closed on incomplete, changed, corrupt or incompatible checkpoints."""
    parameters = _parameters(start, end, entry_cutoff)
    try:
        with gzip.open(filepath, "rt", encoding="utf-8") as handle:
            result = json.load(handle)
        checksum = result.pop("payload_sha256")
        if checksum != _digest(result):
            raise ValueError("Tape payload hash differs")
        result["payload_sha256"] = checksum
        if (
            result["complete"] is not True
            or result["format_version"] != FORMAT_VERSION
            or result["parameters"] != parameters
            or result["start"] != parameters["start"]
            or result["end"] != parameters["end"]
            or result["cutoff"] != parameters["entry_cutoff"]
        ):
            raise ValueError("Tape completeness, version or parameters differ")
        _symbol(result["symbol"])
        if result["code_hashes"] != _code_hashes():
            raise ValueError("Tape implementation hash differs")
        source_path = Path(source_path)
        if (
            result["source"]["sha256"] != _sha(source_path)
            or result["source"]["size_bytes"] != source_path.stat().st_size
        ):
            raise ValueError("Tape source hash differs")
        return result
    except (KeyError, TypeError, OSError, EOFError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid tape checkpoint: {filepath}") from exc


def _job(job):
    return build_tape(**job)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--outputDIR", "--output", dest="output", type=Path, required=True
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 50:
        parser.error("--workers must be between 1 and 50")
    manifest = json.loads(args.manifest.read_text())
    symbols = manifest["symbols"]
    if (
        not isinstance(symbols, list)
        or not symbols
        or len(set(symbols)) != len(symbols)
    ):
        raise ValueError("Manifest symbols must be a nonempty unique list")
    for symbol in symbols:
        _symbol(symbol)
    parameters = _parameters(
        manifest["start"], manifest["end"], manifest["entry_cutoff"]
    )
    paths = {}
    for symbol in symbols:
        path = Path(manifest["data_paths"][symbol])
        paths[symbol] = str(
            (path if path.is_absolute() else args.manifest.parent / path).resolve()
        )
    spec = dict(
        symbols=symbols,
        data_paths=paths,
        parameters=parameters,
        code_hashes=_code_hashes(),
        format_version=FORMAT_VERSION,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "manifest.json"
    if checkpoint.exists():
        if json.loads(checkpoint.read_text()) != spec:
            raise ValueError(
                "Output manifest or implementation differs; use a new directory"
            )
    else:
        if any(args.output.iterdir()):
            raise ValueError("Existing output directory has no matching manifest")
        _publish(checkpoint, spec, compressed=False)
    pending, summaries = [], {}

    def completed(symbol, tape, resumed):
        summaries[symbol] = dict(
            ready_ids=len(tape["ready_ids"]),
            ready_observations=tape["ready_observation_count"],
            elapsed_seconds=tape["elapsed_seconds"],
            resumed=resumed,
        )
        print(_json(dict(symbol=symbol, **summaries[symbol])), flush=True)

    for symbol in symbols:
        output = args.output / f"{symbol}.json.gz"
        times = {k: parameters[k] for k in ("start", "end", "entry_cutoff")}
        if output.exists():
            tape = load_tape(output, paths[symbol], **times)
            if tape["symbol"] != symbol:
                raise ValueError("Checkpoint symbol differs")
            completed(symbol, tape, True)
        else:
            pending.append(
                dict(
                    source_path=paths[symbol],
                    symbol=symbol,
                    output_path=str(output),
                    **times,
                )
            )
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_job, job): job["symbol"] for job in pending}
            for future in as_completed(futures):
                completed(futures[future], future.result(), False)
    print(_json(dict(complete=True, symbols=len(summaries))), flush=True)
    return summaries


if __name__ == "__main__":
    main()
