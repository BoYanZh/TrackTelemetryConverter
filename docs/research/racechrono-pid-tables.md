# RaceChrono PID tables: app names, units, decode formulas

Date: 2026-10-14

## Conclusion

- The app package carries exact display names for every standard PID plus unit classes, raw-byte decode formulas, and a custom-CAN path. Five corrections/additions landed from it: `10058`->Fuel Level (was wrongly Analog 1), `1020`->Fuel Volume, `1003`->Clutch Pos, `14`->Chassis Yaw Rate, `43`->G Force Combined.
- Bulk-adding dormant mappings was REJECTED: file units differ from display units (proof: generic wheel speed 1015 displays km/h while sibling custom PIDs store m/s), so unobserved PIDs cannot be scaled safely. Only corpus-observed PIDs enter `RCZ_PID_MAP`.
- Vehicle-custom CAN PIDs (7-digit) appear nowhere in the app: they come from user CAN configuration at runtime and are unmappable by construction. Unknowns stay raw `OBD_<pid>` with zero data loss.

## Provenance

- Every entry below was cross-checked against recorded sessions before entering the map; see Evidence.

## How to read the app tables

- `map10[pid]` = display name; `map2[pid]` = canonical short id (`d51.j` returns its SECOND key, verified by no-collision).
- `tu.a`/`tu.b` give a unit CLASS per pid; `ia1` resolves it through user metric/imperial prefs into `map11`. The Metric column below is the metric-default resolution.
- `gy2(min, scale, ...)` builds raw-byte formulas as `value = (raw + min) / scale` (e.g. RPM raw/4, intake raw-40, throttle raw/2.55). These decode RAW OBD bytes, not file values: device-12 files already hold engineering values.
- `g40` decodes the custom-CAN path instead: manifold gets +101 kPa barometric compensation, brake is clamped to 0-100 %. No corpus file exercises this path (zero pid-10027 files).

## Evidence (landed mappings)

| pid | app name | file values observed | our mapping |
|---|---|---|---|
| 10058 | Fuel level | x2.00 vs pid 1020 in all 10 files with both (50 L tank) | Fuel Level, %, x1.0 |
| 1020 | fuel_level_vol | -2 ~ 29 (litres) in 12 files | Fuel Volume, L, x1.0 |
| 1003 | clutch_pos | 0 ~ 100 in 3 files | Clutch Pos, %, x1.0 |
| 14 | Z rate of rotation | +/-53, sole yaw source in 2021-22 files | Chassis Yaw Rate, deg/s, x1.0 |
| 43 | Combined acceleration | 0 ~ 19.7 m/s^2 in 6 files | G Force Combined, G, /9.80665 |

## Full reference (generated from v10.2.4 tables)

Status: `map` = in RCZ_PID_MAP (scale shown), `raw` = intentionally unmapped (custom/unknown, preserved as OBD_&lt;pid&gt;), `-` = absent from corpus (dormant by policy).

| pid | app name | short id | unit class | metric unit | raw formula | status |
|---|---|---|---|---|---|---|
| 0 |  |  |  |  |  | - |
| 1 | Time | timestamp | 31 | None |  | - |
| 2 | Distance | distance_traveled | 2 | km |  | - |
| 3 | Position | position | 21 | deg |  | - |
| 4 | Speed | speed | 5 | km/h | (raw+0)/3.6 | map km/h x3.6 |
| 5 | Altitude | altitude | 3 | m |  | - |
| 6 | Bearing | bearing | 11 | deg |  | - |
| 7 | Lateral acceleration | lateral_acc | 17 | G |  | map G x0.10197162129779283 |
| 8 | Longitudinal acceleration | longitudinal_acc | 17 | G |  | map G x0.10197162129779283 |
| 9 | X acceleration | x_acc | 17 | G |  | - |
| 10 | Y acceleration | y_acc | 17 | G |  | - |
| 11 | Z acceleration | z_acc | 17 | G |  | - |
| 12 | X rate of rotation | x_rate_of_rotation | 22 | None |  | - |
| 13 | Y rate of rotation | y_rate_of_rotation | 22 | None |  | - |
| 14 | Z rate of rotation | z_rate_of_rotation | 22 | None |  | map deg/s x1.0 |
| 15 | Device update rate | device_update_rate | 18 | None |  | - |
| 16 | Vertical speed | vertical_speed | 5 | km/h |  | - |
| 17 | Device battery time to empty | device_battery_time_to_empty | 1 | s |  | - |
| 18 | Media capacity | device_media_capacity | 23 | GB |  | - |
| 19 | Media free space | device_media_free_space | 23 | GB |  | - |
| 20 | Elapsed time | elapsed_time | 1 | s |  | - |
| 24 | X rotation vector | x_rotation_vector | 19 | None |  | - |
| 25 | Y rotation vector | y_rotation_vector | 19 | None |  | - |
| 26 | Z rotation vector | z_rotation_vector | 19 | None |  | - |
| 27 | Scalar rotation vector | scalar_rotation_vector | 19 | None |  | - |
| 28 | X magnetic field | x_magnetic_field | 27 | None |  | - |
| 29 | Y magnetic field | y_magnetic_field | 27 | None |  | - |
| 30 | Z magnetic field | z_magnetic_field | 27 | None |  | - |
| 37 | Lean angle | lean_angle | 11 | deg |  | - |
| 38 | Pitch | pitch | 11 | deg |  | - |
| 39 | Yaw | yaw | 11 | deg |  | - |
| 41 | Heart rate | heart_rate | 28 | None |  | - |
| 43 | Combined acceleration | combined_acc | 17 | G |  | map G x0.10197162129779283 |
| 44 | Latitude | latitude | 21 | deg |  | - |
| 45 | Longitude | longitude | 21 | deg |  | - |
| 46 | Device battery level | device_battery_level | 6 | % |  | - |
| 47 | Device battery voltage | device_battery_voltage | 14 | None |  | - |
| 48 | Roll | roll | 11 | deg |  | - |
| 49 | Roll rate | roll_rate | 22 | None |  | - |
| 50 | Pitch rate | pitch_rate | 22 | None |  | - |
| 51 | Yaw rate | yaw_rate | 22 | None |  | map deg/s x1.0 |
| 52 | X rotation angle | x_rotation | 11 | deg |  | - |
| 53 | Y rotation angle | y_rotation | 11 | deg |  | - |
| 54 | Z rotation angle | z_rotation | 11 | deg |  | - |
| 55 | Drift angle | drift_angle | 11 | deg |  | - |
| 1001 | Steering angle | steering_angle | 11 | deg |  | map deg x-1.0 |
| 1002 | Brake position | brake_pos | 6 | % |  | map % x1.0 |
| 1003 | Clutch position | clutch_pos | 6 | % |  | map % x1.0 |
| 1004 | Gear | gear | 20 | None |  | map  x1.0 |
| 1005 | Gearbox temperature | gearbox_temp | 7 | .C |  | map C x1.0 |
| 1006 | Clutch temperature | clutch_temp | 7 | .C |  | - |
| 1007 | Engine oil pressure | engine_oil_pressure | 10 | kPa |  | map kPa x1.0 |
| 1008 | Engine oil level | engine_oil_level | 6 | % |  | - |
| 1009 | Brake temperature | brake_temp | 7 | .C |  | - |
| 1010 | Odometer | odometer | 2 | km |  | - |
| 1011 | Coolant pressure | coolant_pressure | 10 | kPa |  | - |
| 1012 | Throttle angle | throttle_angle | 11 | deg |  | - |
| 1013 | Tyre temperature | tyre_temp | 7 | .C |  | - |
| 1014 | Tyre pressure | tyre_pressure | 10 | kPa |  | - |
| 1015 | Wheel speed | wheel_speed | 5 | km/h |  | - |
| 1016 | Wheel speed | wheel_speed_freq | 18 | None |  | - |
| 1017 | Suspension travel | suspension_travel | 4 | mm |  | - |
| 1019 | Emergency brake position | emergency_brake_pos | 6 | % |  | - |
| 1020 | Fuel level | fuel_level_vol | 32 | None |  | map L x1.0 |
| 1021 | Coolant level | coolant_level | 6 | % |  | - |
| 1022 | Coolant level | coolant_level_vol | 32 | None |  | - |
| 1023 | Air/fuel ratio | air_fuel_ratio | 29 | None |  | - |
| 1024 | Engine oil level | engine_oil_level_vol | 32 | None |  | - |
| 1025 | Knock correction | knock_correction | 11 | deg |  | - |
| 1026 | Wastegate duty cycle | wastegate_duty_cycle | 6 | % |  | - |
| 1027 | Engine power | engine_power | 34 | kW |  | - |
| 1028 | Engine current | engine_current | 39 | None |  | - |
| 1029 | Engine torque | engine_torque | 40 | None |  | - |
| 1030 | Inverter power | inverter_power | 34 | kW |  | - |
| 1031 | Battery level | battery_level_energy | 38 | Nm |  | - |
| 1032 | Battery current | battery_current | 39 | None |  | - |
| 1033 | Brake pressure | brake_pressure | 10 | kPa |  | map kPa x1.0 |
| 1034 | State of charge | state_of_charge | 6 | % |  | - |
| 1035 | Tyre slip | tyre_slip | 6 | % |  | - |
| 1036 | Brake temperature | brake_temp_pct | 6 | % |  | - |
| 1037 | Battery temperature | battery_temp | 7 | .C |  | - |
| 1038 | Battery temperature | battery_temp_pct | 6 | % |  | - |
| 1039 | Inverter temperature | inverter_temp | 7 | .C |  | - |
| 1040 | Inverter temperature | inverter_temp_pct | 6 | % |  | - |
| 1041 | CAT temperature | cat_temp | 7 | .C |  | - |
| 1042 | Battery level | battery_level | 6 | % |  | - |
| 1043 | Battery voltage | battery_voltage | 14 | None |  | - |
| 5000 | Analog | analog | 19 | None |  | - |
| 5001 | Digital | digital | 20 | None |  | - |
| 5002 | Frequency | frequency | 18 | None |  | - |
| 5003 | Temperature | temperature | 7 | .C |  | - |
| 5004 | Acceleration | acceleration | 17 | G |  | - |
| 5005 | Pressure | pressure | 10 | kPa |  | - |
| 5006 | Duration | duration | 1 | s |  | - |
| 5007 | Distance | distance_short | 4 | mm |  | - |
| 5008 | Distance | distance_medium | 3 | m |  | - |
| 5009 | Distance | distance_long | 2 | km |  | - |
| 5010 | Angle | angle | 11 | deg |  | - |
| 5011 | Percent | percent | 6 | % |  | - |
| 5012 | Voltage | voltage | 14 | None |  | - |
| 5015 | Current | current | 39 | None |  | - |
| 5016 | Power | power | 34 | kW |  | - |
| 5017 | Torque | torque | 40 | None |  | - |
| 5018 | Energy | energy | 38 | Nm |  | - |
| 10024 | RPM | rpm | 8 | rpm | (raw+0)/4 | map rpm x1.0 |
| 10025 | Throttle position | throttle_pos | 6 | % | (raw+0)/2.55 | map % x1.0 |
| 10026 | Coolant temperature | coolant_temp | 7 | .C | (raw-40)/1 | map C x1.0 |
| 10027 | Manifold pressure | manifold_pressure | 10 | kPa | (raw+0)/1 | map kPa x1.0 |
| 10028 | Timing advance | timing_advance | 11 | deg | (raw-128)/2 | - |
| 10029 | Intake temperature | intake_temp | 7 | .C | (raw-40)/1 | map C x1.0 |
| 10030 | Airflow rate | airflow_rate | 12 | g/s | (raw+0)/100 | - |
| 10031 | Air temperature | air_temp | 7 | .C | (raw-40)/1 | map C x1.0 |
| 10032 | Engine load | engine_load | 6 | % | (raw+0)/2.55 | - |
| 10033 | Fuel trim short B1 | fuel_trim_short_b1 | 6 | % | (raw-128)/1.28 | - |
| 10034 | Fuel trim long B1 | fuel_trim_long_b1 | 6 | % | (raw-128)/1.28 | - |
| 10035 | Fuel trim short B2 | fuel_trim_short_b2 | 6 | % | (raw-128)/1.28 | - |
| 10036 | Fuel trim long B2 | fuel_trim_long_b2 | 6 | % | (raw-128)/1.28 | - |
| 10037 | Fuel pressure | fuel_pressure | 10 | kPa | (raw+0)/0.333 | - |
| 10038 | Secondary air status | secondary_air_status | 13 | None | (raw+0)/1 | - |
| 10039 | Oxygen sensors | oxygen_sensors | 13 | None | (raw+0)/1 | - |
| 10040 | Vehicle standards | vehicle_standards | 13 | None | (raw+0)/1 | - |
| 10041 | Oxygen sensors 2 | oxygen_sensors_b | 13 | None | (raw+0)/1 | - |
| 10042 | Auxillary input status | auxillary_input_status | 13 | None | (raw+0)/1 | - |
| 10043 | Run time since start | run_time_since_start | 1 | s | (raw+0)/1 | - |
| 10044 | Oxygen sensor 1 B1 | oxygen_sensor_1_b1 | 14 | None | (raw+0)/200 | - |
| 10045 | Oxygen sensor 2 B1 | oxygen_sensor_2_b1 | 14 | None | (raw+0)/200 | - |
| 10046 | Oxygen sensor 3 B1 | oxygen_sensor_3_b1 | 14 | None | (raw+0)/200 | - |
| 10047 | Oxygen sensor 4 B1 | oxygen_sensor_4_b1 | 14 | None | (raw+0)/200 | - |
| 10048 | Oxygen sensor 1 B2 | oxygen_sensor_1_b2 | 14 | None | (raw+0)/200 | - |
| 10049 | Oxygen sensor 2 B2 | oxygen_sensor_2_b2 | 14 | None | (raw+0)/200 | - |
| 10050 | Oxygen sensor 3 B2 | oxygen_sensor_3_b2 | 14 | None | (raw+0)/200 | - |
| 10051 | Oxygen sensor 4 B2 | oxygen_sensor_4_b2 | 14 | None | (raw+0)/200 | - |
| 10052 | Distance with MIL | distance_with_mil | 2 | km | (raw+0)/0.001 | - |
| 10053 | Fuel rail pressure | fuel_rail_pressure | 10 | kPa | (raw+0)/12.66 | - |
| 10054 | Diesel rail pressure | diesel_rail_pressure | 10 | kPa | (raw+0)/0.1 | - |
| 10055 | Commander EGR | commander_egr | 6 | % | (raw+0)/2.55 | - |
| 10056 | EGR error | egr_error | 6 | % | (raw-128)/1.28 | - |
| 10057 | Evoporative purge | evoporative_purge | 6 | % | (raw+0)/2.55 | - |
| 10058 | Fuel level | fuel_level | 6 | % | (raw+0)/2.55 | map % x1.0 |
| 10059 | Warm-ups since clear | warm_ups_since_clear | 13 | None | (raw+0)/1 | - |
| 10060 | Distance since clear | distance_since_clear | 2 | km | (raw+0)/0.001 | - |
| 10061 | Evaporator pressure | evaporator_pressure | 35 | kPa |  | - |
| 10062 | Barometric pressure | barometric_pressure | 10 | kPa | (raw+0)/1 | - |
| 10063 | ECU voltage | ecu_voltage | 14 | None | (raw+0)/1000 | - |
| 10064 | Absolute load | absolute_load | 6 | % | (raw+0)/2.55 | - |
| 10065 | Relative throttle position | relative_throttle_pos | 6 | % | (raw+0)/2.55 | - |
| 10066 | Engine oil temperature | engine_oil_temp | 7 | .C | (raw-40)/1 | map C x1.0 |
| 10067 | Relative accelerator position | relative_accelerator_pos | 6 | % | (raw+0)/2.55 | - |
| 10068 | Engine fuel rate | engine_fuel_rate | 26 | None | (raw+0)/20 | - |
| 10069 | Absolute throttle position B | absolute_throttle_pos_b | 6 | % | (raw+0)/2.55 | - |
| 10070 | Absolute throttle position C | absolute_throttle_pos_c | 6 | % | (raw+0)/2.55 | - |
| 10071 | Accelerator position | accelerator_pos | 6 | % | (raw+0)/2.55 | map % x1.0 |
| 10072 | Accelerator position E | accelerator_pos_e | 6 | % | (raw+0)/2.55 | - |
| 10073 | Accelerator position F | accelerator_pos_f | 6 | % | (raw+0)/2.55 | - |
| 10074 | Air/fuel lambda | air_fuel_lambda | 29 | None | (raw+0)/32768 | - |
| 10075 | Manifold relative pressure | manifold_relative_pressure | 10 | kPa |  | - |
| 10077 | Ethanol fuel content | ethanol_fuel_content | 6 | % | (raw+0)/2.55 | - |
| 20002 | Digital 1/RPM | digital1 | 8 | rpm |  | - |
| 20003 | Analog 1 | analog1 | 19 | None |  | - |
| 20004 | Analog 2 | analog2 | 19 | None |  | - |
| 20005 | Analog 3 | analog3 | 19 | None |  | - |
| 20006 | Analog 4 | analog4 | 19 | None |  | - |
| 20007 | Analog 5 | analog5 | 19 | None |  | - |
| 20010 | Digital 2 | digital2 | 20 | None |  | - |
| 20011 | Analog 6 | analog6 | 19 | None |  | - |
| 20012 | Analog 7 | analog7 | 19 | None |  | - |
| 20013 | Analog 8 | analog8 | 19 | None |  | - |
| 20014 | Analog 9 | analog9 | 19 | None |  | - |
| 20015 | Analog 10 | analog10 | 19 | None |  | - |
| 20016 | Analog 11 | analog11 | 19 | None |  | - |
| 20017 | Analog 12 | analog12 | 19 | None |  | - |
| 20018 | Analog 13 | analog13 | 19 | None |  | - |
| 20019 | Analog 14 | analog14 | 19 | None |  | - |
| 20020 | Analog 15 | analog15 | 19 | None |  | - |
| 30001 | Distance (direct) | distance_direct | 2 | km |  | - |
| 30002 | Satellites | satellites | 16 | None |  | - |
| 30003 | Fix type | fix_type | 24 | None |  | - |
| 30004 | Coordinate precision | coordinate_precision | 15 | None |  | - |
| 30005 | Altitude precision | altitude_precision | 15 | None |  | - |
| 30006 | 3D precision | 3d_precision | 15 | None |  | - |
| 30007 | Accuracy | accuracy | 3 | m |  | - |
| 32001 | Comparison lap time | comparison_lap_time | 41 | s |  | - |
| 32002 | Comparison lap number | comparison_lap_number | 36 | None |  | - |
| 32003 | Previous lap time | previous_lap_time | 41 | s |  | - |
| 32005 | Previous lap number | previous_lap_number | 36 | None |  | - |
| 32012 | Current lap time | lap_time | 1 | s |  | - |
| 32013 | Current lap distance | lap_distance | 2 | km |  | - |
| 32014 | Current lap number | lap_number | 36 | None |  | - |
| 32015 | Current sector time | sector_time | 1 | s |  | - |
| 32016 | Current sector distance | sector_distance | 2 | km |  | - |
| 32024 | Total race time | total_race_time | 1 | s |  | - |
| 32025 | Total race distance | total_race_distance | 2 | km |  | - |
| 32026 | Best lap time | best_lap_time | 41 | s |  | - |
| 32027 | Best lap number | best_lap_number | 36 | None |  | - |
| 32030 |  | lap_time_gain | 30 | s/km |  | - |

## Open work

- Manifold +101 kPa applies on the custom-CAN path only; no corpus file carries pid 10027, so the standard-path scale 1.0 stands unchallenged.
- IMU int32 file scale (x1e-4 = G) is empirical from output value ranges; no code-level origin confirmed.
- Device-12 and device-4 share `RCZ_PID_MAP` by numeric pid. No corpus device-4 file collides with the new entries; the `rcz_dev4_pid_overrides` mechanism exists for future divergences.
- Table pinned to v10.2.4; firmware drift risk on app updates.
- Future-PID workflow: census (`channel2_12_*` PIDs across corpus) -> look up above -> value-check ranges/units against this table -> map + synthetic-RCZ test. Never map unobserved PIDs.
