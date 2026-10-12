"""Venue-name normalization for MoTeC log metadata."""
from __future__ import annotations


def normalize_venue(name):
    if not name:
        return name
    s_lower = name.lower().replace("_", " ").replace("-", " ").replace(",", " ")

    # 1. Thunderhill Raceway Park variants
    if any(k in s_lower for k in ["thunderhill", "thunder hill", "thill", "thunderhil"]):
        if any(k in s_lower for k in ["5 mile", "5 miles", "5mi"]):
            if "double bypass" in s_lower or "db" in s_lower:
                return "Thunderhill 5 Mile Double Bypass"
            elif "west" in s_lower and "bypass" in s_lower:
                return "Thunderhill 5 Mile West Bypass"
            elif "bypass" in s_lower:
                return "Thunderhill 5 Mile Bypass"
            else:
                return "Thunderhill 5 Mile Full"

        if "west" in s_lower:
            if "bypass" in s_lower:
                return "Thunderhill West Bypass"
            # NOTE: check "ccw" before "cw" since "ccw" contains "cw".
            if "ccw" in s_lower:
                return "Thunderhill West CCW"
            if "cw" in s_lower:
                return "Thunderhill West CW"
            return "Thunderhill West"

        if "cyclone" in s_lower:
            return "Thunderhill East Cyclone"

        # If not cyclone/west/5mile, default to Thunderhill East Bypass
        return "Thunderhill East Bypass"

    # 2. Buttonwillow Raceway Park variants
    if "buttonwillow" in s_lower or "bwc" in s_lower:
        if "25ccw" in s_lower or "25 ccw" in s_lower:
            return "Buttonwillow 25CCW"
        elif "13cw" in s_lower or "13 cw" in s_lower:
            return "Buttonwillow 13CW"
        elif "1accw" in s_lower or "1a ccw" in s_lower:
            return "Buttonwillow 1A CCW"
        elif "1acw" in s_lower or "1a cw" in s_lower:
            return "Buttonwillow 1A CW"
        elif "1ccw" in s_lower or "1 ccw" in s_lower:
            return "Buttonwillow 1 CCW"
        elif "1cw" in s_lower or "1 cw" in s_lower:
            return "Buttonwillow 1 CW"
        elif "the circuit" in s_lower or "circuit" in s_lower:
            return "Buttonwillow The Circuit"
        return "Buttonwillow Raceway Park"

    # 3. Laguna Seca
    if "laguna" in s_lower or "seca" in s_lower:
        return "WeatherTech Raceway Laguna Seca"

    # 4. Sonoma Raceway
    if "sonoma" in s_lower:
        return "Sonoma Raceway"

    # 5. Willow Springs
    if "streets of willow" in s_lower or "sow" in s_lower:
        return "Streets of Willow"
    if "willow" in s_lower:
        return "Willow Springs Raceway"

    # 6. Chuckwalla
    if "chuckwalla" in s_lower:
        return "Chuckwalla Valley Raceway"

    # 7. Highway 9 / CA-9
    if any(k in s_lower for k in ["ca 9", "ca9", "highway 9", "hwy 9", "saratoga"]):
        if any(k in s_lower for k in [" n", "_n", "-n", "north"]):
            return "CA-9 N"
        return "CA-9 S"

    return name


#: Canonical venue names produced by normalize_venue(). Keep in sync with its
#: branches; used to decide whether a venue string actually identifies a track.
KNOWN_VENUES = frozenset({
    "Thunderhill 5 Mile Double Bypass",
    "Thunderhill 5 Mile West Bypass",
    "Thunderhill 5 Mile Bypass",
    "Thunderhill 5 Mile Full",
    "Thunderhill West Bypass",
    "Thunderhill West CCW",
    "Thunderhill West CW",
    "Thunderhill West",
    "Thunderhill East Cyclone",
    "Thunderhill East Bypass",
    "Buttonwillow 25CCW",
    "Buttonwillow 13CW",
    "Buttonwillow 1A CCW",
    "Buttonwillow 1A CW",
    "Buttonwillow 1 CCW",
    "Buttonwillow 1 CW",
    "Buttonwillow The Circuit",
    "Buttonwillow Raceway Park",
    "WeatherTech Raceway Laguna Seca",
    "Sonoma Raceway",
    "Streets of Willow",
    "Willow Springs Raceway",
    "Chuckwalla Valley Raceway",
    "CA-9 N",
    "CA-9 S",
})

#: GPS areas for venue fallback detection: (venue, center lat, center lon,
#: radius km). ONE area per facility on purpose: configs sharing tarmac
#: (Thunderhill East/West/Cyclone, Buttonwillow Classic/Circuit, the Willow
#: family) have near-identical centroids, so overlapping per-config areas
#: would make nearest-match a coin flip. GPS therefore only ever claims
#: facility level; config disambiguation stays P1 work. Centers are session
#: GPS medians measured from real logs; observed sessions stay within 1 km
#: of center while known facilities sit 50+ km apart, so 3 km is conservative.
#: Only corpus-verified facilities are listed; files from other tracks keep
#: resolving through filename/metadata strings.
TRACK_AREAS = (
    ("Thunderhill Raceway Park", 39.53993, -122.33150, 3.0),
    ("WeatherTech Raceway Laguna Seca", 36.58544, -121.75438, 3.0),
    ("Sonoma Raceway", 38.16186, -122.45644, 3.0),
    ("Buttonwillow Raceway Park", 35.49767, -119.55506, 3.0),
)

_MIN_GPS_SAMPLES = 10


def is_meaningful_venue(name):
    """Return True when a venue string identifies a known track.

    Anything normalize_venue() passes through unchanged (empty strings,
    AiM free text like "new track 01", generic "Track Day") counts as
    meaningless and is eligible for GPS fallback detection.
    """
    if not name:
        return False
    return normalize_venue(name) in KNOWN_VENUES


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    from math import asin, cos, radians, sin, sqrt

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0 * asin(sqrt(a))


def detect_venue_from_gps(latitudes, longitudes):
    """Return the canonical venue whose area contains the GPS median.

    Returns "" when GPS is missing, too sparse, non-finite, or outside every
    known area. Median (not mean) resists paddock/drive-to-track outliers.
    """
    from statistics import median

    try:
        lats = [float(v) for v in latitudes]
        lons = [float(v) for v in longitudes]
    except (TypeError, ValueError):
        return ""
    import math

    pairs = [(la, lo) for la, lo in zip(lats, lons, strict=False)
             if math.isfinite(la) and math.isfinite(lo)]
    if len(pairs) < _MIN_GPS_SAMPLES:
        return ""
    mlat = median(la for la, _ in pairs)
    mlon = median(lo for _, lo in pairs)
    best = None
    for venue, clat, clon, radius_km in TRACK_AREAS:
        dist = _haversine_km(mlat, mlon, clat, clon)
        if dist <= radius_km and (best is None or dist < best[0]):
            best = (dist, venue)
    return best[1] if best else ""


