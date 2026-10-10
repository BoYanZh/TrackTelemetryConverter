"""Manual regression gate over the local telemetry corpus (NOT run in CI).

Walks a directory of real log files (default: data/), converts each one
with the installed CLI, and compares structural stats against a baseline.
Catches parser regressions on real-world files that fixtures don't cover.

Privacy design (read before changing this file):
- Only aggregate STRUCTURAL stats are recorded: detected type, channel
  count, duration, lap count, and a hash of the sorted channel list.
- No sample values, coordinates, timestamps, metadata strings, or file
  names leave the machine. Baseline keys are SHA-256 of relative paths.
- The baseline lives NEXT TO the data (data/.corpus_baseline.json,
  git-ignored with the rest of data/). Nothing here is committed.
- Conversion outputs go to a system temp dir and are deleted afterwards.
- Console output prints short hashes and drift details locally only.

Usage (from the repo root, package installed):
  python tools/corpus_check.py [--dir data] [--bless] [--timeout 300]

  --bless  (re)write the baseline instead of checking against it. Bless
           only after manually confirming the current outputs are correct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

BASELINE_NAME = ".corpus_baseline.json"
# Extensions this project's CLI writes that are not inputs themselves.
SKIP_EXTENSIONS = {".ld", ".ldx"}
# A sibling .dbc (same directory) is passed for CAN-detected files.
DBC_EXTENSION = ".dbc"


def _find_candidate_files(root):
    """Yield convertible input paths under root, skipping our own outputs."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if os.path.splitext(name)[1].lower() in SKIP_EXTENSIONS:
                continue
            yield os.path.join(dirpath, name)


def _sibling_dbc(path):
    """Return a .dbc next to the file when one exists (for CAN logs)."""
    directory = os.path.dirname(path)
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return None
    for name in names:
        if name.lower().endswith(DBC_EXTENSION):
            return os.path.join(directory, name)
    return None


def _convert(path, out_dir, timeout, dbc=None):
    """Convert one file with the CLI; return the exit code."""
    out = os.path.join(out_dir, "out.ld")
    args = [sys.executable, "-m", "track_telemetry_converter",
            path, "AUTO", "--output", out, "--force"]
    if dbc:
        args += ["--dbc", dbc]
    try:
        proc = subprocess.run(  # noqa: S603 - list-form, no shell; running our own CLI is the point
            args, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124
    return proc.returncode


def _collect_stats(path):
    """Parse the input in-process; return structural stats (no values)."""
    from track_telemetry_converter.log import DataLog
    from track_telemetry_converter.parsers import auto_detect_log_type

    log_type = auto_detect_log_type(path)
    log = DataLog()
    if log_type == "CAN":
        return {"type": log_type, "parse": "skipped-can"}
    parse = getattr(log, f"from_{log_type.lower()}_log", None)
    if parse is None:
        return {"type": log_type, "parse": "no-parser"}
    with open(path, encoding="utf-8", errors="ignore") as handle:
        text = handle.read()
    if log_type in ("RCZ", "DLAP", "IBT", "XRK", "FIT"):
        parse(path)
    else:
        parse(text.splitlines())
    names = sorted(log.channels)
    digest = hashlib.sha256(",".join(names).encode("utf-8")).hexdigest()[:16]
    laps_info = getattr(log, "laps_info", None) or {}
    return {
        "type": log_type,
        "parse": "ok",
        "channels": len(names),
        "duration": round(log.duration(), 1),
        "laps": laps_info.get("total_laps", 0),
        "chans_hash": digest,
    }


def _scan(root, timeout):
    """Convert every candidate and return {key: record} keyed by path hash."""
    records = {}
    with tempfile.TemporaryDirectory(prefix="corpus_check_") as tmp:
        for path in _find_candidate_files(root):
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            key = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16]
            rc = _convert(path, tmp, timeout, _sibling_dbc(path))
            record: dict = {"rc": rc}
            if rc == 0:
                try:
                    record.update(_collect_stats(path))
                except Exception as exc:  # noqa: BLE001 - stats must never break the gate
                    record["parse"] = f"error: {type(exc).__name__}"
            records[key] = record
    return records


def _load_baseline(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _diff(baseline, current):
    """Split current records into (new, drifted, ok) against the baseline."""
    new, drifted, ok = {}, {}, {}
    for key, record in current.items():
        old = baseline.get(key)
        if old is None:
            new[key] = record
        elif old != record:
            drifted[key] = {"was": old, "now": record}
        else:
            ok[key] = record
    gone = {k: v for k, v in baseline.items() if k not in current}
    return new, drifted, ok, gone


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data",
                        help="Corpus root (default: data/)")
    parser.add_argument("--bless", action="store_true",
                        help="(Re)write the baseline instead of checking")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-file conversion timeout in seconds")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        print(f"ERROR: corpus directory not found: {root}")
        return 2
    baseline_path = os.path.join(root, BASELINE_NAME)

    print(f"Scanning {root} ...")
    current = _scan(root, args.timeout)
    print(f"Converted {len(current)} files.")

    if args.bless:
        with open(baseline_path, "w", encoding="utf-8") as handle:
            json.dump(current, handle, indent=2, sort_keys=True)
        print(f"Baseline written to {baseline_path} ({len(current)} entries).")
        return 0

    baseline = _load_baseline(baseline_path)
    if baseline is None:
        print(f"No baseline at {baseline_path}; run once with --bless first.")
        return 2
    new, drifted, ok, gone = _diff(baseline, current)
    print(f"{len(ok)} unchanged, {len(new)} new, "
          f"{len(drifted)} drifted, {len(gone)} gone.")
    for key in sorted(new):
        print(f"  NEW     {key} {new[key]}")
    for key in sorted(drifted):
        print(f"  DRIFT   {key} was={drifted[key]['was']} now={drifted[key]['now']}")
    for key in sorted(gone):
        print(f"  GONE    {key} {gone[key]}")
    return 1 if (drifted or gone) else 0


if __name__ == "__main__":
    sys.exit(main())
