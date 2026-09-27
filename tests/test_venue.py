"""Tests for venue-name normalization (metadata.normalize_venue)."""

from track_telemetry_converter.metadata import normalize_venue


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
