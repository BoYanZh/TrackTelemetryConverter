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
    for venue, clat, clon, _ in TRACK_AREAS:
        lats = [clat + 0.001 * (i % 5) for i in range(20)]
        lons = [clon - 0.001 * (i % 5) for i in range(20)]
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
    assert _resolve_venue_name(_args(), log) == "Thunderhill East Bypass"

    log = _gps_log(35.49767, -119.55506)
    assert _resolve_venue_name(_args(), log) == "Buttonwillow Raceway Park"


def test_resolve_venue_never_overrides_explicit_or_known():
    from track_telemetry_converter.cli import _resolve_venue_name

    log = _gps_log(39.53993, -122.33150)
    assert _resolve_venue_name(_args(venue_name="My Backyard"), log) == "My Backyard"

    log = _gps_log(35.49767, -119.55506, venue="Laguna Seca")
    assert _resolve_venue_name(_args(), log) == "WeatherTech Raceway Laguna Seca"

    log = _gps_log(0.0, 0.0)
    assert _resolve_venue_name(_args(), log) == "new track 01"
