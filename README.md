# TrackTelemetryConverter

> Formerly named `MotecLogGenerator` (repository/project name).

A single Python CLI for converting motorsports telemetry into verified MoTeC
`.ld` and `.ldx` files.

Supported inputs:

- iRacing `.ibt`
- Assetto Corsa ACTI `.ld` (Car Coord X/Y placed by track profile; `--track` selects, `--anchor` adds custom profiles)
- real-world MoTeC `.ld` logger files (explicit `MOTEC_LD`, or single-file `AUTO`; ACTI sim `.ld` stays on the ACTI path)
- RaceChrono `.rcz` and CSV
- AIM `.xrk` / `.xrz`
- Garmin `.fit`
- Racelogic VBOX `.vbo`
- Dragy `.dlap` (requires `pip install ".[dlap]"`)
- Tesla Track Mode CSV (`telemetry-v1-*.csv`)
- TrackAddict / RaceRender-Data CSV
- COBB Accessport and PB Buddy CSV
- raw CAN logs with a DBC file
- generic CSV

The converter normalizes common channel names and units, resamples channels,
adds generic derived channels, preserves source channels such as RCZ ECU pitch,
and writes `.ld`/`.ldx` through a verified temporary-file pipeline. Existing
outputs are never replaced unless `--force` is supplied, and the input file is
never used as an output target.

## Install

Python 3.10 through 3.14 is supported.

```bash
python -m pip install .
track-telemetry --help
```

Install only the optional parsers you need:

```bash
python -m pip install ".[can]"
python -m pip install ".[fit]"
python -m pip install ".[xrk]"
python -m pip install ".[dlap]"
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
track-telemetry dragy.dlap DLAP
track-telemetry can.log CAN --dbc vehicle.dbc
```

The package module is an equivalent fallback:

```bash
python -m track_telemetry_converter session.rcz AUTO
```

`AUTO` detects IBT, RCZ, XRK/XRZ, FIT, VBO, DLAP, Tesla, TrackAddict, PB Buddy, AIM/RaceStudio/RaceRender CSV,
Accessport CSV, generic CSV, and MoTeC LD inputs (single-file mode; directory
batch skips `.ld` so the converter never re-ingests its own outputs).
Explicit choices are `CAN`, `CSV`,
`ACCESSPORT`, `RACECHRONO`, `RCZ`, `PBBUDDY`, `TRACKADDICT`, `TESLA`, `VBO`, `DLAP`, `IBT`, `XRK`, `FIT`, `ACTI`, and `MOTEC_LD`.

Common options:

```bash
track-telemetry session.rcz AUTO --output converted.ld
track-telemetry session.rcz AUTO --output converted.ld --force
track-telemetry session.rcz AUTO --csv
track-telemetry session.fit AUTO --csv --csv-wallclock
track-telemetry session.rcz AUTO --g-source sensor --frequency 25
```

Inspect a file without creating MoTeC outputs. The command writes a versioned,
privacy-safe JSON report to stdout. It omits sample values, GPS coordinates,
session metadata, and the input path. Use `--debug-report` to also save the
same report; `--debug-report` requires `--inspect`, existing reports require
`--force`, and the input cannot be used as the report path. Channel frequency
uses the median positive timestamp interval. Gaps count intervals over one
second. Validation rejects empty channels and nonfinite, duplicate, or
decreasing timestamps; nonfinite values are reported as warnings.

```bash
track-telemetry session.rcz AUTO --inspect
track-telemetry backup.rcz RCZ --inspect --session session_20260101_1000
track-telemetry session.rcz RCZ --inspect --debug-report diagnostics.json
```

For RCZ backups, choose a session ID. If that session contains multiple stints,
choose one with `--stint N`. Each channel's `usable_samples` counts aligned
time/value pairs where both fields are finite and reports the minimum and
maximum usable times. When array lengths differ, only overlapping indices can
form pairs. Inspect does not resample channels or derive conversion channels.
Malformed inputs still produce JSON with a stable error code and a nonzero exit
status.

`frequency_hz` describes the parsed timestamp cadence. Some RCZ channels may
already share the GPS time grid, so this value does not necessarily describe a
sensor's native sampling rate.

Python callers can use the same conversion service as the CLI and batch workers:

```python
from track_telemetry_converter.conversion import ConversionOptions, ConversionRequest, convert

result = convert(ConversionRequest(
    source="session.rcz",
    log_type="AUTO",
    options=ConversionOptions(output="converted.ld", frequency="25"),
))
if result.status == "success":
    print(result.output_paths)
else:
    print(result.error)
```

`convert` returns `success`, `skipped` for existing outputs, or `failed`, with
structured errors, ordered messages, and the files actually written. It does
not print progress or exit the process. Successful single-export results include
a diagnostic summary and completed processing stages. RCZ session/stint fanout,
output naming, `force`, and verified LD/LDX writes use the same path as the CLI.
Requests and results can be passed between worker processes; VBO column selections
use the raw `vbo_columns` string option.

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

Batch-convert a whole directory with per-file format detection:

```bash
track-telemetry data/ AUTO --recursive --force
```

`AUTO` detects RCZ, IBT, XRK/XRZ, FIT, VBO, DLAP, and CSV dialects for each file,
so mixed inputs convert in one run. `.ld` files are skipped in directory mode
(they are the converter's own outputs, except ACTI sim logs handled by an
explicit ACTI batch); convert a real-world `.ld` with single-file
`MOTEC_LD` or single-file `AUTO`. Outputs are written next to each input and
existing outputs are skipped unless `--force` is given. Add `--recursive` to
include subdirectories. RaceChrono backup archives are skipped with a notice
unless `--session all` is passed, in which case each archive expands under its
own `<backup>_sessions/` directory. `--output`, `--session ID`, `--stint`, and
`--lap` cannot be combined with directory mode.

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
- RCZ CAN/OBD and IMU channels use their recorded device timestamps. Values
  outside each sensor's first/last sample are unavailable (`NaN`) rather than
  repeated from an endpoint, even without `--mask-interp-gaps`. That flag
  additionally masks long gaps *inside* the recorded interval.
- `--min-lap-sec` controls the minimum reconstructed RCZ out/timed/in segment
  duration; the legacy `--min_lap_sec` spelling remains accepted.
- An RCZ leading segment that starts at 5 km/h or faster is retained as a
  `Partial Out Lap`, with a warning that the source recording began mid-lap.
- `.ld` and `.ldx` are staged, parsed back, and committed as a pair.
- `--force` is required to replace existing generated files.
- `--g-source {auto,sensor,calc}` selects hardware or GPS-derived G channels.
- `--mask-interp-gaps` leaves long sampling gaps as `NaN` instead of bridging
  them.
- Venue falls back to GPS-area detection when filename/metadata name no
  known track (e.g. AiM free text like "new track 01"); an explicit
  `--venue_name` is never overridden.
- `--csv`, `--vbo`, `--dlap`, `--gpx`, and `--kml` add optional exports.
- `--vbo-columns` selects a subset of VBO extra columns, or renames them
  (`CustomName=Channel Name`); defaults to all available extras.

## Reading individual IBT laps in Python

```python
from track_telemetry_converter.log import DataLog

log = DataLog()
log.from_ibt_log("session.ibt", raw_channels=["LapDistPct", "SessionTime", "YawRate"])
for lap in log.iter_laps(channels=["Ground Speed", "Brake Pos", "Throttle Pos"]):
    print(dict(lap.info))

# lap_num is unique within this recording, even when iRacing resets its counter.
lap = log.get_lap(2, channels=["Ground Speed", "Steering Angle"])
speed = lap.channels.get("Ground Speed")
if speed is not None:
    print(speed.timestamps, speed.values, speed.units, speed.sample_rate)
    print(dict(speed.provenance))
native_distance = lap.raw_channels.get("LapDistPct")
```

`iter_laps()` and `get_lap()` return read-only views of existing arrays without
resampling or rebasing each lap. Missing requested channels are omitted; an
unknown `lap_num` raises `KeyError`. Each `info` includes elapsed start/end times,
duration, `stint`, original `source_lap_num` when available, and `complete`.
First/last segments and segments interrupted by a reset remain partial.
`valid` is separate from completeness: `None` means the available telemetry
does not establish official lap validity. Duration alone never establishes it.
These views share storage with the log; they are not detached snapshots.
Extract before explicitly calling `resample()` if you need native samples.
The canonical `Lap Number` channel retains its historical crossing count,
starting at zero. It differs from the unique segment `lap_num` used by this API.
If no complete eligible lap exists, `fastest_lap` is `None`.

Elapsed timestamps use recorded `SessionTime` intervals, starting at zero.
Backward, repeated, or nonfinite intervals use one tick interval instead;
`metadata["ibt"]["time_repaired"]` records this repair when SessionTime exists.
`source_timestamps` retains the original SessionTime values for alignment and
inspection. Without SessionTime, timestamps use the header tick rate. Timing
boundaries have sample-level precision and are not interpolated across the
start/finish line. Resets and session changes split stints, so laps cannot span
them. Missing or contradictory lap signals can leave segments partial.
`LapDistPct` wraps corroborate counter increments when present. Distance jumps
of more than half a lap also split continuity, including active resets where
the counter keeps advancing. Smaller teleports without another reset signal
may remain undetected. Sparse recordings and configurations with separate
start/finish lines, such as Tourist, may conservatively yield partial laps.
This detector does not claim exact official timing or official lap validity.

`raw_channels=None` keeps the existing canonical channel selection. Pass a
list of original IBT names, a single name, or `"all"` to populate
`log.raw_channels`. This separate namespace never feeds automatic MoTeC export.
Raw values retain their dtype, units, and array width; scalar shapes are
`(ticks,)` and array shapes are `(ticks, count)`. Character variables use `S1`
bytes and booleans use NumPy bools. `provenance` records the native type code,
count, `count_as_time`, and description. Arrays marked `count_as_time` retain
their recorded rows without guessing sub-sample timestamps; `sample_rate`
is the row tick rate. `metadata["ibt"]["variables"]` catalogs valid variable
headers even when no raw values are requested. Unsupported type codes and
headers outside a tick's byte bounds are skipped. Raw views share the file's
in-memory byte buffer; selecting all variables does not copy every array.
For `count_as_time` arrays, provenance also reports `subsample_rate` as tick
rate times count. This preserves the SDK's distinction between array elements
and [time subdivisions](https://www.iracing.com/2017-season-1-release-notes/).

Speed, steering, yaw rate, acceleration, and pressure retain the existing
canonical conversions. Native `LapDistPct`, coordinates, and source timestamps
are available for downstream distance-based comparison. This API does not
align laps, adjudicate iRacing incident penalties, or provide coaching.

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
