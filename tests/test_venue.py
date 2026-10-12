"""Tests for venue-name normalization (metadata.normalize_venue)."""

from types import SimpleNamespace

from track_telemetry_converter.log import DataLog
from track_telemetry_converter.metadata import (
    KNOWN_VENUES,
    TRACK_AREAS,
    detect_venue_from_gps,
    is_meaningful_venue,
    normalize_venue,
)


def test_buttonwillow_config_variants():
    assert normalize_venue("Buttonwillow 13 CW") == "Buttonwillow 13CW"
    assert normalize_venue("Buttonwillow 13CW") == "Buttonwillow 13CW"
    assert normalize_venue("buttonwillow_13_cw_1") == "Buttonwillow 13CW"
    assert normalize_venue("Buttonwillow 1A CCW") == "Buttonwillow 1A CCW"
    assert normalize_venue("buttonwillow_1a_ccw_1") == "Buttonwillow 1A CCW"
    assert normalize_venue("Buttonwillow 1 CCW") == "Buttonwillow 1 CCW"
    assert normalize_venue("Buttonwillow 25 CCW") == "Buttonwillow 25CCW"
    assert normalize_venue("Buttonwillow 25CCW") == "Buttonwillow 25CCW"
    assert normalize_venue("Buttonwillow Circuit") == "Buttonwillow The Circuit"
    assert normalize_venue("Buttonwillow The Circuit") == "Buttonwillow The Circuit"
    # Plain Buttonwillow keeps the generic park name.
    assert normalize_venue("Buttonwillow") == "Buttonwillow Raceway Park"


def test_thunderhill_config_variants():
    assert normalize_venue("Thunderhill 5 Mile Double Bypass") == "Thunderhill 5 Mile Double Bypass"
    assert normalize_venue("Thunderhill 5 Mile Full") == "Thunderhill 5 Mile Full"
    assert normalize_venue("Thunderhill 5 Mile West Bypass") == "Thunderhill 5 Mile West Bypass"
    assert normalize_venue("Thunderhill East Bypass") == "Thunderhill East Bypass"
    assert normalize_venue("thunder_hill_east_bypass") == "Thunderhill East Bypass"
    assert normalize_venue("Thunderhill East Cyclone") == "Thunderhill East Cyclone"
    assert normalize_venue("Thunderhill West CCW") == "Thunderhill West CCW"
    assert normalize_venue("Thunderhill West CW") == "Thunderhill West CW"
    assert normalize_venue("Thunderhill West Bypass") == "Thunderhill West Bypass"
    assert normalize_venue("Thunderhill West") == "Thunderhill West"


def test_other_tracks():
    assert normalize_venue("Laguna Seca") == "WeatherTech Raceway Laguna Seca"
    assert normalize_venue("Sonoma") == "Sonoma Raceway"


def test_unknown_names_pass_through():
    assert normalize_venue("My Backyard Kart Track") == "My Backyard Kart Track"
    assert normalize_venue("") == ""
    assert normalize_venue(None) is None


def test_known_venues_stay_in_sync():
    for venue in KNOWN_VENUES:
        assert normalize_venue(venue) == venue


def test_is_meaningful_venue():
    assert not is_meaningful_venue("")
    assert not is_meaningful_venue(None)
    assert not is_meaningful_venue("new track 01")
    assert not is_meaningful_venue("aim_session_5")
    assert not is_meaningful_venue("Track Day")
    assert is_meaningful_venue("Thunderhill East Bypass")
    assert is_meaningful_venue("thunder_hill_east_bypass")
    assert is_meaningful_venue("WeatherTech Raceway Laguna Seca")


def test_detect_venue_from_gps():
    for venue, clat, clon, radius_km in TRACK_AREAS:
        step = 0.001 if radius_km >= 1.0 else 0.0002
        lats = [clat + step * (i % 5) for i in range(20)]
        lons = [clon - step * (i % 5) for i in range(20)]
        assert detect_venue_from_gps(lats, lons) == venue
    assert detect_venue_from_gps([40.7] * 20, [-74.0] * 20) == ""
    assert detect_venue_from_gps([], []) == ""
    assert detect_venue_from_gps([39.5] * 5, [-122.3] * 5) == ""


def _args(**overrides):
    base = {
        "log": "aim_session_5.csv",
        "venue_name": "",
        "event_name": "",
        "driver": "",
        "vehicle_id": "",
        "vehicle_weight": 0,
        "vehicle_type": "",
        "vehicle_comment": "",
        "event_session": "",
        "long_comment": "",
        "short_comment": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _gps_log(lat, lon, venue="new track 01", samples=20):
    from track_telemetry_converter.channels import CH_GPS_LATITUDE, CH_GPS_LONGITUDE

    log = DataLog()
    log.metadata["venue_name"] = venue
    log.add_channel(CH_GPS_LATITUDE, "deg", float, 7)
    log.add_channel(CH_GPS_LONGITUDE, "deg", float, 7)
    log.channels[CH_GPS_LATITUDE].set_samples(
        [float(i) for i in range(samples)], [lat] * samples,
    )
    log.channels[CH_GPS_LONGITUDE].set_samples(
        [float(i) for i in range(samples)], [lon] * samples,
    )
    return log


def test_resolve_venue_prefers_gps_over_meaningless_strings():
    from track_telemetry_converter.cli import _resolve_venue_name

    log = _gps_log(39.53993, -122.33150)
    assert _resolve_venue_name(_args(), log) == "Thunderhill Raceway Park"

    log = _gps_log(35.48920, -119.54440)
    assert _resolve_venue_name(_args(), log) == "Buttonwillow Raceway Park"

    log = _gps_log(35.49760, -119.55430)
    assert _resolve_venue_name(_args(), log) == "Buttonwillow The Circuit"


def test_gps_separates_thunderhill_east_from_west():
    # Measured session medians: 3 East Bypass + Cyclone + 5-mile all fall in
    # the East cluster (same tarmac / union course) -> park name only.
    assert detect_venue_from_gps([39.53976] * 20, [-122.33214] * 20) == "Thunderhill Raceway Park"
    assert detect_venue_from_gps([39.53947] * 20, [-122.33133] * 20) == "Thunderhill Raceway Park"
    assert detect_venue_from_gps([39.53944] * 20, [-122.33237] * 20) == "Thunderhill Raceway Park"
    # West cluster -> "Thunderhill West" (sub-config detail stays with strings).
    assert detect_venue_from_gps([39.53688] * 20, [-122.33858] * 20) == "Thunderhill West"
    assert detect_venue_from_gps([39.53720] * 20, [-122.33865] * 20) == "Thunderhill West"
    # Midpoint in the gap between the areas: safe passthrough, no coin flip.
    assert detect_venue_from_gps([39.53827] * 20, [-122.33534] * 20) == ""


def test_gps_separates_buttonwillow_classic_from_circuit():
    # Measured session medians (4 Classic configs, 2 Circuit sessions).
    assert detect_venue_from_gps([35.48917] * 20, [-119.54445] * 20) == "Buttonwillow Raceway Park"
    assert detect_venue_from_gps([35.48911] * 20, [-119.54576] * 20) == "Buttonwillow Raceway Park"
    assert detect_venue_from_gps([35.49767] * 20, [-119.55506] * 20) == "Buttonwillow The Circuit"
    assert detect_venue_from_gps([35.49754] * 20, [-119.55354] * 20) == "Buttonwillow The Circuit"
    # Midpoint in the gap between the areas: safe passthrough, no coin flip.
    assert detect_venue_from_gps([35.49340] * 20, [-119.54935] * 20) == ""


def test_thunderhill_west_gets_shared_beacon_traps():
    from track_telemetry_converter.models import Message

    log = DataLog()
    log.metadata["venue_name"] = "Thunderhill West"
    log.add_channel("GPS Latitude", "deg", float, 7)
    log.add_channel("GPS Longitude", "deg", float, 7)
    lats, lons, stamps = [], [], []
    for t in range(0, 26):
        stamps.append(float(t))
        lats.append(39.537866)
        lons.append(-122.334007 + (t - 21) * 0.0001)
    for t in range(66, 81):
        stamps.append(float(t))
        lats.append(39.537866)
        lons.append(-122.334007 + (t - 71) * 0.0001)
    log.channels["GPS Latitude"].messages = [
        Message(t, v) for t, v in zip(stamps, lats, strict=False)
    ]
    log.channels["GPS Longitude"].messages = [
        Message(t, v) for t, v in zip(stamps, lons, strict=False)
    ]
    beacons = log.detect_beacons()
    assert len(beacons) == 2
    assert all(name == "Start/Finish" for _, name in beacons)


def test_resolve_venue_never_overrides_explicit_or_known():
    from track_telemetry_converter.cli import _resolve_venue_name

    log = _gps_log(39.53993, -122.33150)
    assert _resolve_venue_name(_args(venue_name="My Backyard"), log) == "My Backyard"

    log = _gps_log(35.49767, -119.55506, venue="Laguna Seca")
    assert _resolve_venue_name(_args(), log) == "WeatherTech Raceway Laguna Seca"

    log = _gps_log(0.0, 0.0)
    assert _resolve_venue_name(_args(), log) == "new track 01"
