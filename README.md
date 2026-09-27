# TrackTelemetryConverter

> Formerly named `MotecLogGenerator` (repository/project name).

A single Python CLI for converting motorsports telemetry into verified MoTeC
`.ld` and `.ldx` files.

Supported inputs:

- iRacing `.ibt`
- RaceChrono `.rcz` and CSV
- AIM `.xrk` / `.xrz`
- Garmin `.fit`
- Racelogic VBOX `.vbo`
- COBB Accessport and PB Buddy CSV
- raw CAN logs with a DBC file
- generic CSV

The converter normalizes common channel names and units, resamples channels,
adds generic derived channels, preserves source channels such as RCZ ECU pitch,
and writes `.ld`/`.ldx` through a verified temporary-file pipeline. Existing
outputs are never replaced unless `--force` is supplied, and the input file is
never used as an output target.

## Install

Python 3.8 through 3.14 is supported. AIM XRK/XRZ import requires Python 3.10
or newer because `libxrk` does not publish Python 3.8/3.9 builds; all other
input formats remain available on Python 3.8/3.9.

```bash
python -m pip install .
track-telemetry --help
```

Install only the optional parsers you need:

```bash
python -m pip install ".[can]"
python -m pip install ".[fit]"
python -m pip install ".[xrk]"
```

For every optional parser and the test suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Use

The installed command is the primary interface:

```bash
track-telemetry session.ibt AUTO
track-telemetry session.rcz RCZ
track-telemetry session.xrk XRK
track-telemetry session.fit FIT
track-telemetry vbox.vbo VBO
track-telemetry can.log CAN --dbc vehicle.dbc
```

The package module is an equivalent fallback:

```bash
python -m track_telemetry_converter session.rcz AUTO
```

`AUTO` detects IBT, RCZ, XRK/XRZ, FIT, VBO, PB Buddy, AIM/RaceChrono CSV,
Accessport CSV, and generic CSV inputs. Explicit choices are `CAN`, `CSV`,
`ACCESSPORT`, `RACECHRONO`, `RCZ`, `PBBUDDY`, `VBO`, `IBT`, `XRK`, and `FIT`.

Common options:

```bash
track-telemetry session.rcz AUTO --output converted.ld
track-telemetry session.rcz AUTO --output converted.ld --force
track-telemetry session.rcz AUTO --csv
track-telemetry session.fit AUTO --csv --csv-wallclock
track-telemetry session.rcz AUTO --g-source sensor --frequency 25
```

RaceChrono multi-session backups require an explicit selection:

```bash
track-telemetry backup.rcz RCZ --list-sessions
track-telemetry backup.rcz RCZ --session session_20260101_1000
track-telemetry backup.rcz RCZ --session all
track-telemetry backup.rcz RCZ --session all --output-dir converted_sessions
```

A selected session uses `<backup>_<session-id>.ld/.ldx` by default. Sessions
with multiple stints add `_stintN`. `--session all` writes one verified pair per
stint under `<backup>_sessions/` unless `--output-dir` is provided; it cannot be
combined with `--output` or a specific `--stint`.

Run `track-telemetry --help` for the authoritative option list.

## Behavior

- RCZ sessions with multiple stints are split automatically when `--stint all`
  is used.
- RaceChrono backup archives are detected without extraction. They produce no
  output until `--session ID` or `--session all` is supplied; `--list-sessions`
  reports the session ID, track, duration, lap count, and stints.
- RCZ `--lap N` selects the reconstructed lap number within a stint and rebases
  that single-lap export to zero elapsed time.
- RCZ timestamp streams are decoded as full 64-bit Unix milliseconds. Channel
  timestamps remain zero-based seconds for MoTeC compatibility, while
  `DataLog.time_origin_epoch_ms` and `DataLog.datetime_utc` preserve the exact
  absolute origin for callers that need wall-clock alignment.
- `--min-lap-sec` controls the minimum reconstructed RCZ out/timed/in segment
  duration; the legacy `--min_lap_sec` spelling remains accepted.
- An RCZ leading segment that starts at 5 km/h or faster is retained as a
  `Partial Out Lap`, with a warning that the source recording began mid-lap.
- `.ld` and `.ldx` are staged, parsed back, and committed as a pair.
- `--force` is required to replace existing generated files.
- `--g-source {auto,sensor,calc}` selects hardware or GPS-derived G channels.
- `--mask-interp-gaps` leaves long sampling gaps as `NaN` instead of bridging
  them.
- `--csv`, `--gpx`, and `--kml` add optional exports.

## Project layout

```text
src/track_telemetry_converter/   application package and CLI
tests/                     regression tests and telemetry fixtures
pyproject.toml             package, dependency, and CLI configuration
```

The repository intentionally has no second application, compatibility script,
or standalone analysis-tool layer. Parsing, conversion, and export behavior is
owned by the `track-telemetry` application.

## License

This project is licensed under GPL-3.0-only; see [LICENSE](LICENSE).

The application vendors the GPL-3.0
[`gotzl/ldparser`](https://github.com/gotzl/ldparser) implementation used for
MoTeC LD binary parsing/writing. Its license is retained alongside the vendored
source at `src/track_telemetry_converter/_vendor/LDParser.LICENSE`.

MoTeC and i2 are trademarks of their respective owner. This project is an
independent telemetry conversion utility and does not replace licensed MoTeC
hardware or software.
