"""
Centralized Constants & Channel Name Mapping Definitions for TrackTelemetryConverter.

Provides a Single Source of Truth for:
  - Canonical MoTeC channel names and standard units.
  - Multi-source channel alias mappings (iRacing, RaceChrono, AiM Solo, PB Buddy, VBOX, Cobb, Generic CSVs).
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Canonical MoTeC Channel Names (Single Source of Truth)
# ----------------------------------------------------------------------------
CH_GROUND_SPEED         = "Ground Speed"
CH_VEHICLE_SPEED        = "Vehicle Speed"
CH_CG_ACCEL_LAT         = "CG Accel Lateral"
CH_CG_ACCEL_LON         = "CG Accel Longitudinal"
CH_CG_ACCEL_LAT_SMOOTH  = "CG Accel Lateral Smooth"
CH_CG_ACCEL_LON_SMOOTH  = "CG Accel Long Smooth"
CH_GPS_LATITUDE         = "GPS Latitude"
CH_GPS_LONGITUDE        = "GPS Longitude"
CH_GPS_HEADING          = "GPS Heading"
CH_GPS_ALTITUDE         = "GPS Altitude"
CH_GPS_SATS             = "GPS Satellites"
CH_GPS_ACCURACY         = "GPS Accuracy"
CH_GPS_FIX              = "GPS Fix"
CH_LAP_NUMBER           = "Lap Number"
CH_RUNNING_TIME         = "Running Time"
CH_CORR_DIST            = "Corr Dist"
CH_DEVICE_BATTERY       = "Device Battery"
CH_THROTTLE_POS         = "Throttle Pos"
CH_ACCELERATOR_POS      = "Accelerator Pos"
CH_BRAKE_PRESS          = "Brake Press"
CH_BRAKE_POS            = "Brake Pos"
CH_CLUTCH_POS           = "Clutch Pos"
CH_ENGINE_RPM           = "Engine RPM"
CH_STEERING_ANGLE       = "Steering Angle"
CH_COOLANT_TEMP         = "Coolant Temp"
CH_ENGINE_OIL_TEMP      = "Engine Oil Temp"
CH_ENGINE_OIL_PRESS     = "Engine Oil Press"
CH_GEAR                 = "Gear"
CH_GEARBOX_TEMP         = "Gearbox Temp"
CH_YAW_RATE             = "Chassis Yaw Rate"
CH_LEAN_ANGLE           = "Lean Angle"
CH_INTAKE_TEMP          = "Intake Temp"
CH_MANIFOLD_PRESS       = "Manifold Press"
CH_AMBIENT_TEMP         = "Ambient Temp"
CH_ROLL_ANGLE           = "Roll Angle"
CH_PITCH_ANGLE          = "Pitch Angle"
CH_G_COMBINED           = "G Force Combined"
CH_WHEEL_SPEED_AVG      = "Wheel Speed Avg"
CH_WHEEL_SPEED_FL       = "Wheel Speed FL"
CH_WHEEL_SPEED_FR       = "Wheel Speed FR"
CH_WHEEL_SPEED_RL       = "Wheel Speed RL"
CH_WHEEL_SPEED_RR       = "Wheel Speed RR"
CH_BRAKE_PRESS_FL       = "Brake Press FL"
CH_BRAKE_PRESS_FR       = "Brake Press FR"
CH_BRAKE_PRESS_RL       = "Brake Press RL"
CH_BRAKE_PRESS_RR       = "Brake Press RR"
CH_TIRE_PRESS_FL        = "Tire Press FL"
CH_TIRE_PRESS_FR        = "Tire Press FR"
CH_TIRE_PRESS_RL        = "Tire Press RL"
CH_TIRE_PRESS_RR        = "Tire Press RR"
CH_FUEL_LEVEL           = "Fuel Level"
CH_BATTERY_VOLTAGE      = "Battery Voltage"
CH_TRACK_TEMP           = "Track Temp"
CH_LAP_DISTANCE         = "Lap Distance"

DISCRETE_CHANNELS = {CH_GEAR, CH_LAP_NUMBER, CH_GPS_FIX, CH_GPS_SATS}

# ----------------------------------------------------------------------------
# Channel Alias Mappings: Raw header string -> (canonical_name, unit, dec)
# ----------------------------------------------------------------------------
CHANNEL_ALIASES = {
    # Speed
    "Ground Speed":           (CH_GROUND_SPEED, "km/h", 2),
    "GPS Speed":              (CH_GROUND_SPEED, "km/h", 2),
    "Speed(km/h)":            (CH_GROUND_SPEED, "km/h", 2),
    "Speed":                  (CH_GROUND_SPEED, "km/h", 2),
    "velocity":               (CH_GROUND_SPEED, "km/h", 2),
    "velocity-calc":          (CH_GROUND_SPEED, "km/h", 2),
    "velocity kmh":           (CH_GROUND_SPEED, "km/h", 2),
    "Vehicle Speed":          (CH_VEHICLE_SPEED, "km/h", 2),
    "velocity-canbus":        (CH_VEHICLE_SPEED, "km/h", 2),

    # GPS Coordinates
    "GPS Latitude":           (CH_GPS_LATITUDE, "deg", 7),
    "GPS Longitude":          (CH_GPS_LONGITUDE, "deg", 7),
    "Latitude":               (CH_GPS_LATITUDE, "deg", 7),
    "Longitude":              (CH_GPS_LONGITUDE, "deg", 7),
    "lat":                    (CH_GPS_LATITUDE, "deg", 7),
    "long":                   (CH_GPS_LONGITUDE, "deg", 7),

    # Accelerations
    "CG Accel Lateral":       (CH_CG_ACCEL_LAT, "G", 4),
    "GPS_LatAcc":             (CH_CG_ACCEL_LAT, "G", 4),
    "LateralAcc":             (CH_CG_ACCEL_LAT, "G", 4),
    "lateral_acc":            (CH_CG_ACCEL_LAT, "G", 4),
    "latacc":                 (CH_CG_ACCEL_LAT, "G", 4),
    "latacc-calc":            (CH_CG_ACCEL_LAT, "G", 4),
    "lat accel g":            (CH_CG_ACCEL_LAT, "G", 4),
    "CG Accel Longitudinal":  (CH_CG_ACCEL_LON, "G", 4),
    "GPS_LongAcc":            (CH_CG_ACCEL_LON, "G", 4),
    "LineAcc":                (CH_CG_ACCEL_LON, "G", 4),
    "LongAcc":                (CH_CG_ACCEL_LON, "G", 4),
    "longitudinal_acc":       (CH_CG_ACCEL_LON, "G", 4),
    "longacc":                (CH_CG_ACCEL_LON, "G", 4),
    "longacc-calc":           (CH_CG_ACCEL_LON, "G", 4),
    "long accel g":           (CH_CG_ACCEL_LON, "G", 4),

    # GPS Meta
    "GPS Heading":            (CH_GPS_HEADING, "deg", 2),
    "GPS_Bearing":            (CH_GPS_HEADING, "deg", 2),
    "Bearing":                (CH_GPS_HEADING, "deg", 2),
    "heading":                (CH_GPS_HEADING, "deg", 2),
    "GPS Altitude":           (CH_GPS_ALTITUDE, "m", 2),
    "GPS_Height":             (CH_GPS_ALTITUDE, "m", 2),
    "height":                 (CH_GPS_ALTITUDE, "m", 2),
    "GPS Satellites":         (CH_GPS_SATS, "", 0),
    "sats":                   (CH_GPS_SATS, "", 0),
    "satellites":             (CH_GPS_SATS, "", 0),

    # Lap Info
    "Lap Number":             (CH_LAP_NUMBER, "", 0),
    "Lap":                    (CH_LAP_NUMBER, "", 0),
    "Lap Count":              (CH_LAP_NUMBER, "", 0),
    "lap_number":             (CH_LAP_NUMBER, "", 0),

    # Driver Inputs & ECU
    "Throttle Pos":           (CH_THROTTLE_POS, "%", 2),
    "Throttle Position":      (CH_THROTTLE_POS, "%", 2),
    "PPS":                    (CH_THROTTLE_POS, "%", 2),
    "accelerator_pos":        (CH_THROTTLE_POS, "%", 2),
    "accelerator_pos-canbus": (CH_THROTTLE_POS, "%", 2),
    "Brake Press":            (CH_BRAKE_PRESS, "kPa", 2),
    "Brake Pressure":         (CH_BRAKE_PRESS, "kPa", 2),
    "BrakePress":             (CH_BRAKE_PRESS, "bar", 2),
    "brake_pressure":         (CH_BRAKE_PRESS, "kPa", 2),
    "brake_pressure-canbus":  (CH_BRAKE_PRESS, "kPa", 2),
    "Brake Pos":              (CH_BRAKE_POS, "%", 2),
    "brake_pos":              (CH_BRAKE_POS, "%", 2),
    "brake_pos-canbus":       (CH_BRAKE_POS, "%", 2),
    "Engine RPM":             (CH_ENGINE_RPM, "rpm", 2),
    "RPM":                    (CH_ENGINE_RPM, "rpm", 2),
    "rpm-canbus":             (CH_ENGINE_RPM, "rpm", 2),
    "Steering Angle":         (CH_STEERING_ANGLE, "deg", 2),
    "SteerAngle":             (CH_STEERING_ANGLE, "deg", 2),
    "steering_angle":         (CH_STEERING_ANGLE, "deg", 2),
    "steering_angle-canbus":  (CH_STEERING_ANGLE, "deg", 2),
    "Coolant Temp":           (CH_COOLANT_TEMP, "C", 2),
    "coolant_temp":           (CH_COOLANT_TEMP, "C", 2),
    "coolant_temp-canbus":    (CH_COOLANT_TEMP, "C", 2),
    "Engine Oil Temp":        (CH_ENGINE_OIL_TEMP, "C", 2),
    "engine_oil_temp":        (CH_ENGINE_OIL_TEMP, "C", 2),
    "engine_oil_temp-canbus": (CH_ENGINE_OIL_TEMP, "C", 2),
    "Gear":                   (CH_GEAR, "", 0),
    "gear-canbus":            (CH_GEAR, "", 0),
    "Gearbox Temp":           (CH_GEARBOX_TEMP, "C", 2),
    "gearbox_temp-canbus":    (CH_GEARBOX_TEMP, "C", 2),
    "throttle_pos-canbus":    (CH_THROTTLE_POS, "%", 2),
    "yaw_rate-canbus":        (CH_YAW_RATE, "deg/s", 3),
    "wheel_speed_fl-canbus":  (CH_WHEEL_SPEED_FL, "km/h", 2),
    "wheel_speed_fr-canbus":  (CH_WHEEL_SPEED_FR, "km/h", 2),
    "wheel_speed_rl-canbus":  (CH_WHEEL_SPEED_RL, "km/h", 2),
    "wheel_speed_rr-canbus":  (CH_WHEEL_SPEED_RR, "km/h", 2),

    # Dragy VBO `-obd` columns (also written by this project's VBO exporter).
    "rpm-obd":                 (CH_ENGINE_RPM, "rpm", 0),
    "acceleration-obd":         (CH_ACCELERATOR_POS, "%", 2),
    "brake_pressure-obd":       (CH_BRAKE_PRESS, "kPa", 2),
    "engine_oil_temp-obd":      (CH_ENGINE_OIL_TEMP, "C", 2),
    "coolant_temp-obd":         (CH_COOLANT_TEMP, "C", 2),
    "manifold_pressure-obd":    (CH_MANIFOLD_PRESS, "kPa", 2),

    # RaceChrono raw OBD_ IDs (Toyota GR86 / FT86 CAN Bus)
    # NOTE: the even PIDs are tire pressures (kPa, verified on the
    # 2026-09-13 Buttonwillow session: ~193-262 kPa). The odd PIDs are NOT
    # tire status -- they are the four wheel speeds in m/s (verified on the
    # 2026-10-03/04 Thunder Hill sessions, see RCZ_PID_MAP). The alias table
    # carries no scale factor, so these map to km/h and rely on the consumer
    # (xrk_parser._convert_values handles m/s -> km/h).
    "OBD_164854":             (CH_TIRE_PRESS_FL, "kPa", 2),
    "OBD_132086":             (CH_TIRE_PRESS_FR, "kPa", 2),
    "OBD_197622":             (CH_TIRE_PRESS_RL, "kPa", 2),
    "OBD_99318":              (CH_TIRE_PRESS_RR, "kPa", 2),
    "OBD_164855":             (CH_WHEEL_SPEED_FR, "km/h", 2),
    "OBD_132087":             (CH_WHEEL_SPEED_RL, "km/h", 2),
    "OBD_197623":             (CH_WHEEL_SPEED_FL, "km/h", 2),
    "OBD_99319":              (CH_WHEEL_SPEED_RR, "km/h", 2),
    "OBD_3155786":            (CH_ENGINE_OIL_TEMP, "C", 2),
    "OBD_45098826":           (CH_AMBIENT_TEMP, "C", 2),
    "OBD_1020":               (CH_FUEL_LEVEL, "%", 2),
    # Verified against RaceChrono v10.2.4 channel tables: "Fuel level", unit class %.
    # Corpus: exact x2.00 vs pid 1020 (fuel volume, 50 L tank arithmetic) in
    # all 10 files carrying both. No device-4 10058 exists, so no key collision.
    "OBD_10058":              (CH_FUEL_LEVEL, "%", 2),
    # Raw undecoded OBD fallbacks below are intentionally literal, not canonical channels.
    "OBD_44050250":           ("Digital 42", "", 0),
    "OBD_44045193":           ("Status Flag 44045193", "", 0),
}

# ----------------------------------------------------------------------------
# iRacing .ibt native channel -> MoTeC canonical mapping
# Each tuple: (ibt_var_name, ch_name, units, decimals, convert_fn_or_None)
# ----------------------------------------------------------------------------
_G = 9.80665
IBT_CHANNEL_MAP = [
    ("Speed",              CH_GROUND_SPEED,     "km/h",    2, lambda x: x * 3.6),
    ("Lat",                CH_GPS_LATITUDE,     "deg",     7, None),
    ("Lon",                CH_GPS_LONGITUDE,    "deg",     7, None),
    ("Alt",                CH_GPS_ALTITUDE,     "m",       2, None),
    ("LatAccel",           CH_CG_ACCEL_LAT,     "G",       4, lambda x: x / _G),
    ("LongAccel",          CH_CG_ACCEL_LON,     "G",       4, lambda x: x / _G),
    ("YawRate",            CH_YAW_RATE,         "deg/s",   3, np.degrees),
    ("YawNorth",           CH_GPS_HEADING,      "deg",     2, np.degrees),
    ("SteeringWheelAngle", CH_STEERING_ANGLE,   "deg",     2, np.degrees),
    ("Throttle",           CH_THROTTLE_POS,     "%",       2, lambda x: x * 100.0),
    ("Brake",              CH_BRAKE_POS,        "%",       2, lambda x: x * 100.0),
    ("RPM",                CH_ENGINE_RPM,       "rpm",     0, None),
    ("Gear",               CH_GEAR,             "",        0, None),
    ("WaterTemp",          CH_COOLANT_TEMP,     "C",       2, None),
    ("OilTemp",            CH_ENGINE_OIL_TEMP,  "C",       2, None),
    ("OilPress",           CH_ENGINE_OIL_PRESS, "kPa",     2, lambda x: x * 100.0),
    ("ManifoldPress",      CH_MANIFOLD_PRESS,   "kPa",     2, lambda x: x * 100.0),
    ("FuelLevel",          CH_FUEL_LEVEL,       "l",       2, None),
    ("Voltage",            CH_BATTERY_VOLTAGE,  "V",       2, None),
    ("TrackTemp",          CH_TRACK_TEMP,       "C",       2, None),
    ("LapDistPct",         CH_LAP_DISTANCE,     "%",       2, lambda x: x * 100.0),
]

IBT_WHEEL_SPEED_MAP = [
    ("LFspeed", CH_WHEEL_SPEED_FL),
    ("RFspeed", CH_WHEEL_SPEED_FR),
    ("LRspeed", CH_WHEEL_SPEED_RL),
    ("RRspeed", CH_WHEEL_SPEED_RR),
]

IBT_BRAKE_PRESS_MAP = [
    ("LFbrakeLinePress", CH_BRAKE_PRESS_FL),
    ("RFbrakeLinePress", CH_BRAKE_PRESS_FR),
    ("LRbrakeLinePress", CH_BRAKE_PRESS_RL),
    ("RRbrakeLinePress", CH_BRAKE_PRESS_RR),
]

# ----------------------------------------------------------------------------
# RaceChrono RCZ binary ZIP channel PID -> (canonical_name, unit, scale, offset)
# ----------------------------------------------------------------------------
RCZ_PID_MAP = {
    "10024": (CH_ENGINE_RPM, "rpm", 1.0, 0.0),            # 605 ~ 7489 rpm
    "10071": (CH_ACCELERATOR_POS, "%", 1.0, 0.0),         # 0 ~ 100 % (Accelerator Pedal Pos - 0% when idle)
    "10025": (CH_THROTTLE_POS, "%", 1.0, 0.0),            # Throttle Body Valve Angle
    "1002":  (CH_BRAKE_POS, "%", 1.0, 0.0),               # 0 ~ 100 %
    "1033":  (CH_BRAKE_PRESS, "kPa", 1.0, 0.0),           # 0 ~ 9600 kPa
    "1007":  (CH_ENGINE_OIL_PRESS, "kPa", 1.0, 0.0),      # 110 ~ 683 kPa
    "10066": (CH_ENGINE_OIL_TEMP, "C", 1.0, 0.0),         # 57 ~ 104 C
    "10026": (CH_COOLANT_TEMP, "C", 1.0, 0.0),            # 67 ~ 95 C
    "1005":  (CH_GEARBOX_TEMP, "C", 1.0, 0.0),            # 63 ~ 114 C
    "10029": (CH_INTAKE_TEMP, "C", 1.0, 0.0),             # 29 ~ 42 C
    "10027": (CH_MANIFOLD_PRESS, "kPa", 1.0, 0.0),        # 20 ~ 100 kPa
    "1001":  (CH_STEERING_ANGLE, "deg", -1.0, 0.0),       # -294 ~ 457 deg
    "1004":  (CH_GEAR, "", 1.0, 0.0),                     # -1 ~ 5 (Integer gear)
    "4":     (CH_GROUND_SPEED, "km/h", 3.6, 0.0),         # raw is m/s (0 ~ 50 m/s -> km/h)
    "51":    (CH_YAW_RATE, "deg/s", 1.0, 0.0),            # -37 ~ 39 deg/s
    "7":     (CH_ROLL_ANGLE, "deg", 1.0, 0.0),            # Roll Angle (-11.6 ~ 11.2 deg)
    "8":     (CH_PITCH_ANGLE, "deg", 1.0, 0.0),           # Pitch Angle (-10.6 ~ 4.0 deg)
    "10031": (CH_AMBIENT_TEMP, "C", 1.0, 0.0),            # Ambient Temp (16 ~ 87 C)
    # PIDs verified against RaceChrono v10.2.4 channel tables,
    # each value-checked against corpus sessions:
    # - 10058 "Fuel level" (% class): exact x2.00 vs pid 1020 (fuel volume,
    #   50 L tank arithmetic) in all 10 files carrying both.
    "10058": (CH_FUEL_LEVEL, "%", 1.0, 0.0),
    # - 1003 "clutch_pos" (% class): 0 ~ 100 in 3 files.
    "1003": (CH_CLUTCH_POS, "%", 1.0, 0.0),
    # - 14 "Z rate of rotation": +/-53 value range, sole yaw source in
    #   2021-22 files (no pid 51 / gyro files there); deg/s like pid 51.
    "14": (CH_YAW_RATE, "deg/s", 1.0, 0.0),
    # - 43 "Combined acceleration": 0 ~ 19.7 m/s^2 in 6 files, hence G needs
    #   /9.80665. Shares overwrite semantics with racechrono combined_acc:
    #   calculate_g_sum recomputes it from lat/lon G at math time.
    "43": (CH_G_COMBINED, "G", 1.0 / _G, 0.0),
    # - 1020 "fuel_level_vol": -2 ~ 29 in 12 files, litres of a 50 L tank
    #   (exact x0.50 vs pid 10058 fuel level). Kept distinct from CH_FUEL_LEVEL
    #   so both readings survive; no unit convention for fuel volume exists yet.
    "1020": ("Fuel Volume", "L", 1.0, 0.0),
    "1053576": (CH_WHEEL_SPEED_AVG, "rpm", 1.0, 0.0),     # Wheel Speed (1296 ~ 3230)
    # Four individual wheel speeds, raw unit is m/s (0 ~ 49 m/s).
    # Verified on 2026-10-03/04 Thunder Hill East Bypass sessions (RWD GR86):
    # left = {197623, 132087} vs right = {164855, 99319} from GPS-heading-backed
    # turn analysis; driven (rear) axle = {132087, 99319} from monotonic
    # throttle/brake response (spins up under power, dips under braking).
    "197623": (CH_WHEEL_SPEED_FL, "km/h", 3.6, 0.0),
    "164855": (CH_WHEEL_SPEED_FR, "km/h", 3.6, 0.0),
    "132087": (CH_WHEEL_SPEED_RL, "km/h", 3.6, 0.0),
    "99319":  (CH_WHEEL_SPEED_RR, "km/h", 3.6, 0.0),
    # Tire pressures in kPa (verified value range ~193-262 kPa on the
    # 2026-09-13 Buttonwillow session). Corner assignment below follows the
    # pre-existing CHANNEL_ALIASES guess and is NOT verified -- the sibling
    # wheel-speed PIDs turned out to use a different corner scheme.
    "164854": (CH_TIRE_PRESS_FL, "kPa", 1.0, 0.0),
    "132086": (CH_TIRE_PRESS_FR, "kPa", 1.0, 0.0),
    "197622": (CH_TIRE_PRESS_RL, "kPa", 1.0, 0.0),
    "99318":  (CH_TIRE_PRESS_RR, "kPa", 1.0, 0.0),
}

# ----------------------------------------------------------------------------
# Track Start/Finish Beacons Coordinate Database
# ----------------------------------------------------------------------------
TRACK_BEACONS = {
    "WeatherTech Raceway Laguna Seca": {"lat": 36.58620173, "lon": -121.75661697, "heading_deg": 218.45, "name": "Start/Finish"},
    "Sonoma Raceway": {"lat": 38.16146533, "lon": -122.45478067, "heading_deg": 309.0, "name": "Start/Finish"},
    "Thunderhill East Bypass": {"lat": 39.53786600, "lon": -122.33400700, "heading_deg": 93.0, "name": "Start/Finish"},
    "Thunderhill East Cyclone": {"lat": 39.53786600, "lon": -122.33400700, "heading_deg": 93.0, "name": "Start/Finish"},
    "Thunderhill West": {"lat": 39.53786600, "lon": -122.33400700, "heading_deg": 93.0, "name": "Start/Finish"},
    "Thunderhill 5 Mile Double Bypass": {"lat": 39.53786600, "lon": -122.33400700, "heading_deg": 93.0, "name": "Start/Finish"},
    "Thunderhill 5 Mile Bypass": {"lat": 39.53786600, "lon": -122.33400700, "heading_deg": 93.0, "name": "Start/Finish"},
    "Thunderhill 5 Mile Full": {"lat": 39.53786600, "lon": -122.33400700, "heading_deg": 93.0, "name": "Start/Finish"},
    "CA-9 S": [
        {"lat": 37.25430017, "lon": -122.03870167, "heading_deg": 264.0, "name": "Start"},
        {"lat": 37.25811550, "lon": -122.12268867, "heading_deg": 221.0, "name": "9-35 Junction"},
        {"lat": 37.16793350, "lon": -122.13589900, "heading_deg": 179.0, "name": "Finish"},
    ],
    "CA-9 N": [
        {"lat": 37.16852117, "lon": -122.13586100, "heading_deg": 3.0, "name": "Start"},
        {"lat": 37.25817950, "lon": -122.12256300, "heading_deg": 44.0, "name": "9-35 Junction"},
        {"lat": 37.25431967, "lon": -122.03861317, "heading_deg": 81.0, "name": "Finish"},
    ],
}
