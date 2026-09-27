# libxrk 0.13.0 XRK timestamp semantics

Date: 2026-08-25

## Conclusion

TrackTelemetryConverter should require `libxrk>=0.13`. The timestamp difference is
an intentional GPS timecode bug fix, not an independent rebase of every
channel. Later releases remain eligible and are guarded by the XRK regression
tests and Python-version CI matrix.

## Primary-source evidence

libxrk 0.12.0 reconstructed a non-monotonic GPS stream by adding 65,536 ms
after every decrease. Version 0.13.0 instead uses nearest-band phase unwrapping,
so only a step close to a real 16-bit rollover advances the clock. Smaller
out-of-order steps, replayed blocks, and zero/dropout records retain their
reconstructed time. See the [v0.13.0 release notes](https://github.com/m3rlin45/libxrk/releases/tag/v0.13.0)
and the [implementing commit](https://github.com/m3rlin45/libxrk/commit/d2ef208f21aeda9970ba84c04a81b14a9446ce1b).

The maintainer verified the fix against AIM's DLL with 6,062 samples and
0.000 ms per-sample deviation. Across 353 tested files, 29 were affected. Raw
GPS tables may retain duplicate timestamps and one backwards step to match
AIM, while `get_channels_as_table()` remains monotonic and unique. Version
0.13.0 also adds a name-addressable `lap_type` column; TrackTelemetryConverter does
not depend on the exact positional schema.

## Repository reproduction

The same `tests/fixtures/aim_sample.xrk` file was parsed in isolated Python 3.10
environments with each released dependency version.

| Observation | libxrk 0.12.0 | libxrk 0.13.0 |
|---|---:|---:|
| GPS Latitude range | 4.730-602.951 s | 0.068-598.249 s |
| RPM range | 0.000-598.358 s | 0.000-598.358 s |
| First derived lap | 23.834-86.967 s | 19.132-82.265 s |

Non-GPS timing did not change. The old GPS endpoint exceeded the RPM/logger
endpoint by about 4.6 seconds; the corrected endpoint differs by 0.109 seconds.
TrackTelemetryConverter's `_dedupe_samples` then sorts and deduplicates the raw stream
for export.

The previous regression assertion requiring GPS to start around 4.7 seconds
encoded the old reconstruction. It has been replaced with checks that the GPS
offset remains on the shared session clock, aligns with the logger endpoint,
and has finite, unique, strictly monotonic exported timestamps.

## Uncertainty

This repository has one XRK fixture and cannot run AIM's proprietary DLL
locally. The upstream source change, AIM-DLL comparison, and 353-file corpus
are stronger evidence than preserving the old fixture-specific offset. Future
libxrk versions are intentionally allowed, with CI responsible for detecting
new incompatibilities.
