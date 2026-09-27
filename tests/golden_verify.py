"""Verify refactoring preserves parse/process behavior by comparing against golden snapshot."""
import json
import os
import sys

from track_telemetry_converter.log import DataLog

EXAMPLES = os.path.abspath(os.path.join(os.path.dirname(__file__), "fixtures"))
GOLDEN = os.path.abspath(os.path.join(os.path.dirname(__file__), "golden", "golden.json"))


def _read(filename):
    with open(os.path.join(EXAMPLES, filename), encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def _json_normalize(obj):
    """Normalize Python objects (tuples->lists etc.) to match JSON round-tripped golden."""
    return json.loads(json.dumps(obj, default=str))


def _dump(log):
    channels = {}
    for name, ch in sorted(log.channels.items()):
        vals = [m.value for m in ch.messages]
        channels[name] = {
            "units": ch.units,
            "decimals": ch.decimals,
            "n": len(ch.messages),
            "first": [ch.messages[0].timestamp, ch.messages[0].value] if ch.messages else None,
            "last": [ch.messages[-1].timestamp, ch.messages[-1].value] if ch.messages else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }
    return {
        "channels": channels,
        "metadata": {k: v for k, v in sorted(log.metadata.items())},
        "datetime": log.datetime.isoformat() if log.datetime else None,
        "laps_info": getattr(log, "laps_info", None),
        "traps": getattr(log, "traps", []),
        "duration": log.duration(),
    }


def _process(filename, parser):
    log = DataLog()
    if parser == "from_csv_log":
        log.from_csv_log(_read(filename))
    elif parser == "from_pbbuddy_log":
        log.from_pbbuddy_log(_read(filename))
    elif parser == "from_vbo_log":
        log.from_vbo_log(_read(filename))
    elif parser == "from_ibt_log":
        log.from_ibt_log(os.path.join(EXAMPLES, filename))
    elif parser == "from_racechrono_log":
        log.from_racechrono_log(_read(filename))
    elif parser == "from_accessport_log":
        log.from_accessport_log(_read(filename))
    else:
        raise ValueError(parser)
    return log


def main():
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)

    failed = 0
    total = 0
    for key, expected in sorted(golden.items()):
        filename, parser = key.split("__")
        log = _process(filename, parser)
        parse_dump = _dump(log)
        log.calculate_math_channels(g_source="auto")
        log.resample("auto")
        processed_dump = _dump(log)

        for stage, actual in (("parse", parse_dump), ("processed", processed_dump)):
            exp = _json_normalize(expected[stage])
            actual = _json_normalize(actual)
            total += 1
            if actual == exp:
                print(f"  OK   {key} [{stage}]")
            else:
                failed += 1
                print(f"  FAIL {key} [{stage}]")
                # Find first divergence
                all_keys = set(actual) | set(exp)
                for k in sorted(all_keys):
                    if actual.get(k) != exp.get(k):
                        print(f"    field '{k}':")
                        print(f"      expected: {str(exp.get(k))[:300]}")
                        print(f"      actual:   {str(actual.get(k))[:300]}")
                        break

    print(f"\n{total - failed}/{total} checks passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
