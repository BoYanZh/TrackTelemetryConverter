# RCZ device model: fragment declarations, clock signatures, IMU calibration

Date: 2026-10-09

## Conclusion

RaceChrono device numbers (`100`/`101`/`102`, `200`, …) are **not hardware
IDs** — the same physical receiver appears under different numbers on
different phones/pairings. Each stint's `sessionfragment.json` declares the
mapping explicitly (`primaryGpsDeviceIndex`, `imuAccDevice`,
`imuGyroDevice`), plus per-device mounting calibration (`orientation`
quaternion, `gyroBias`). The parser therefore selects channels by
**declaration first, number pattern as fallback**, applies the declared
gyro bias in the sensor frame, keeps both yaw sources instead of silently
dropping one, and records everything in metadata. The orientation
quaternion is recorded but deliberately **not applied**: rotating
device-frame accel into vehicle frame could not be validated (it degrades
agreement on real data — see below), so the parser stays a faithful reader
and leaves frame alignment to the estimation layer.

## Primary-source evidence

Thunder Hill East Bypass session (`session_20261003_0935_*`, base stint).
All numbers below were measured from that file, not assumed.

### The fragment declares everything

`sessionfragment.json` (one per stint: base plus `resume_1/2/3`):

| key | value |
|---|---|
| `devices.items` | `{100/model121/type1}`, `{101/model121/type2}`, `{102/model121/type3}`, `{200/model405/type12}` |
| `primaryGpsDeviceIndex` | `100` |
| `imuAccDevice` / `imuGyroDevice` | `101` / `102` |
| `devices2[101].orientation` | `[-0.179, -0.231, -0.956, 0.094]` |
| `devices2[102].gyroBias` | `[-2.141, 0.741, 1.052]` (deg/s, device frame) |
| `imuUseForCalcEx` | `true` |

IDs 100/101/102 share model 121: one physical box (GPS + accel + gyro).
ID 200/model 405/type 12 is the second box (car OBD/CAN).

### Clock signatures identify hardware, numbers don't

| family | rate | drift vs GPS clock |
|---|---:|---|
| `1/100` GPS | 25.00 Hz | self |
| `2/101` accel | 25.00 Hz | slope 1.000000, resid p95 0.00 |
| `3/102` gyro | 25.00 Hz | slope 1.000000, resid p95 0.00 |
| `12/200` CAN | 27.85 / 44.09 Hz | own clocks |

GPS + accel + gyro on one zero-drift 25 Hz clock = a single external
receiver. A phone would show ~1 Hz GPS plus an independent IMU rate. This
is the identification rule: **shared clock, not device number**.

### gyroBias semantics verified

Receiver gyro `_12/_13/_14` straight-line means `(-1.63, +0.65, +0.78)`
deg/s vs declared bias `(-2.14, +0.74, +1.05)`: same sign, same axis
order, plausible magnitude (residual = temperature drift + road input).
The parser therefore subtracts the bias per axis in the sensor frame and
records the applied values in `rcz_gyro_bias_applied`. After subtraction
the receiver gyro straight mean drops from +0.77 to −0.20 deg/s.

### Silent gyro overwrite fixed

`channel_3_102_0_14_0` exists in the file, so the receiver gyro parsed
fine — then the CAN PID 51 branch overwrote `Chassis Yaw Rate` with no
membership guard and the receiver data was lost. Now CAN keeps
`Chassis Yaw Rate` (car-mounted, aligned — the better source here) and the
receiver gyro is retained as `Receiver Yaw Rate`, with `yaw_rate_source`
and `yaw_rate_receiver` both recorded. Verified: receiver-vs-CAN
correlation −0.9996 (same signal, mirrored mounting — preserved
as-recorded, sign left to the consumer).

### Orientation recorded, not applied

12 combinations tried (quaternion × conjugate × 6 axis permutations),
scored by sustained-corner agreement between rotated lateral accel and
CAN `v·yaw`. Best 0.0516 G (with wrong scale −0.46 and −0.85 G bias),
worst 0.078 G — a flat landscape with no physical winner, all worse than
the naive mapping (0.045 G, slope +1.03, bias +0.01 G after sign flip).
A pure rotation cannot create the observed bias structure from ~zero-mean
device axes, which suggests RaceChrono already compensates tilt/gravity
in the exported channels, or the axis/file/quaternion conventions differ
from the natural reading. Applying an unvalidated rotation would be
synthesizing data, so the quaternion is exposed in `rcz_imu_orientation`
for the estimation layer instead.

## Parser contract (rcz_parser.py)

- Channel selection: fragment-declared device id first
  (`channel_1_{gps}`, `channel_2_{accel}_0_10_0`, `channel_3_{gyro}_0_14_0`),
  legacy `(300,200,100)` / `(301,201,101)` / `(302,202,102)` probing as
  fallback. Fixtures without a fragment exercise the legacy path
  byte-identically (existing suite).
- Gyro bias subtracted in the sensor frame iff the fragment declares a
  bias for that exact device; axis index from the value-file suffix
  (`12/13/14` → 0/1/2), recorded in metadata.
- Metadata on every fragment-bearing parse: `rcz_device_roles`,
  `rcz_devices`, `rcz_imu_orientation`, `rcz_gyro_bias_declared`,
  `yaw_rate_source`, `yaw_rate_receiver` (when both exist),
  `rcz_gyro_bias_applied` (when applied).

## Uncertainty / open work

- The full 3D axis mapping (file suffix ↔ device axis, quaternion
  component order/direction, vehicle frame convention) is **unresolved by
  driving data alone** — see "Axis-mapping difficulty" below. The
  decisive experiment is a static 6-position tumble test with the actual
  box (gravity per file gives the exact file↔axis map), not more driving.
- For the sideslip use case the full map is unnecessary: lateral-file
  identity, sign, and bias are observable per session from correlations
  (`ay` vs `v·yaw`, straights vs corners). That auto-calibration belongs
  in the consumer (ApexLayer telemetry), not the converter.
- `gyroBias` temperature drift (≈0.5 deg/s on one axis here) means the
  declared bias is a starting point; consumers should still re-estimate
  bias on straights.

## Axis-mapping difficulty

Unknowns, coupled: (a) value-file suffix ↔ device physical axis,
(b) quaternion component order (`[x,y,z,w]` vs `[w,x,y,z]`),
(c) quaternion direction (device→vehicle vs vehicle→device),
(d) vehicle frame convention (x fwd, y left/right, z up/down),
(e) whether RaceChrono pre-applies the orientation to channel values.

That is 6 × 2 × 2 × 2+ hypotheses before even touching (e), while
steady-state cornering only constrains one projection (lateral agreement),
so the objective is flat (0.052–0.078 G across all 12 tested combos) and
cannot identify the true hypothesis. More driving data does not help;
each hypothesis needs an independent anchor:

1. **Tumble test (decisive for (a))**: box static, each face down ~10 s.
   Gravity (≈1 G) appears in exactly one file per position → exact
   file↔axis map, plus bias re-measurement. One afternoon, no driving.
2. **Vendor docs/source (for (b)–(d))**: the quaternion convention is a
   fact about RaceChrono internals, not something to fit.
3. **Mount-rotation A/B (for (e))**: two sessions with the box rotated
   90° between them; if channel semantics follow the box, RaceChrono does
   not pre-apply; if they stay vehicle-aligned, it does.
