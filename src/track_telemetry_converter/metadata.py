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
            elif "bypass" in s_lower:
                return "Thunderhill 5 Mile Bypass"
            else:
                return "Thunderhill 5 Mile Full"

        if "west" in s_lower:
            if "bypass" in s_lower:
                return "Thunderhill West Bypass"
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


