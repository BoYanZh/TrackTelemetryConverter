"""Generate golden parse/process results for refactoring verification.

Run BEFORE refactoring to snapshot reference outputs, then run golden_verify.py
AFTER refactoring to confirm behavior is unchanged.
"""
import json
import os

from track_telemetry_converter.log import DataLog

EXAMPLES = os.path.join(os.path.dirname(__file__), "fixtures")
OUT_DIR = os.path.join(os.path.dirname(__file__), "golden")


def _read(filename):
    with open(os.path.join(EXAMPLES, filename), encoding="utf-8", errors="ignore") as f:
        return f.readlines()


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
    elif parser == "from_can_log":
        return None  # requires cantools - covered separately when available
    else:
        raise ValueError(parser)
    return log


CASES = [
    ("csv_sample.csv", "from_csv_log"),
    ("pbbuddy_sample.csv", "from_pbbuddy_log"),
    ("vbo_sample.vbo", "from_vbo_log"),
    ("ibt_sample.ibt", "from_ibt_log"),
    ("aim_solo_sample.csv", "from_racechrono_log"),
    ("racechrono_sample.csv", "from_racechrono_log"),
    ("accessport_sample.csv", "from_accessport_log"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    combined = {}
    for filename, parser in CASES:
        log = _process(filename, parser)
        if log is None:
            continue
        key = f"{filename}__{parser}"
        parse_dump = _dump(log)

        # processed: apply the standard pipeline stages (math + resample)
        log.calculate_math_channels(g_source="auto")
        log.resample("auto")
        processed_dump = _dump(log)

        combined[key] = {"parse": parse_dump, "processed": processed_dump}
        print(f"  golden {key}: {len(log.channels)} channels")

    out_path = os.path.join(OUT_DIR, "golden.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=1, sort_keys=True, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
