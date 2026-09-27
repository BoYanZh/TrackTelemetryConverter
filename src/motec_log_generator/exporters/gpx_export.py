"""GPS track exports (GPX / KML) for TrackTelemetryConverter."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ..channels import CH_GPS_ALTITUDE, CH_GPS_LATITUDE, CH_GPS_LONGITUDE, CH_GROUND_SPEED
from .xml_utils import indent_xml


def _find_gps_channels(data_log):
    lat = data_log.channels.get(CH_GPS_LATITUDE) or data_log.channels.get("Real GPS Latitude")
    lon = data_log.channels.get(CH_GPS_LONGITUDE) or data_log.channels.get("Real GPS Longitude")
    return lat, lon

def _downsample_step(channel, target_hz=5.0):
    freq = channel.avg_frequency()
    if freq <= target_hz:
        return 1
    return max(1, int(round(freq / target_hz)))

def _iter_gps_trackpoints(data_log, target_hz=5.0):
    lat_chan, lon_chan = _find_gps_channels(data_log)
    if not lat_chan or not lon_chan:
        return
    n = min(len(lat_chan.messages), len(lon_chan.messages))
    if n == 0:
        return
    alt_chan = data_log.channels.get(CH_GPS_ALTITUDE)
    spd_chan = data_log.channels.get(CH_GROUND_SPEED)
    step = _downsample_step(lat_chan, target_hz)
    for i in range(0, n, step):
        lat = lat_chan.messages[i].value
        lon = lon_chan.messages[i].value
        if abs(lat) < 0.001 and abs(lon) < 0.001:
            continue
        alt = alt_chan.messages[i].value if alt_chan and i < len(alt_chan.messages) else 0.0
        spd = spd_chan.messages[i].value if spd_chan and i < len(spd_chan.messages) else None
        yield lat, lon, alt, spd

def write_gpx(motec_log, gpx_filename, data_log):

    points = list(_iter_gps_trackpoints(data_log))
    if not points:
        return False

    root = ET.Element("gpx", {
        "version": "1.1",
        "creator": "TrackTelemetryConverter",
        "xmlns": "http://www.topografix.com/GPX/1/1"
    })
    trk = ET.SubElement(root, "trk")
    ET.SubElement(trk, "name").text = motec_log.venue_name or "Track Session"
    trkseg = ET.SubElement(trk, "trkseg")

    for lat, lon, alt, spd in points:
        pt = ET.SubElement(trkseg, "trkpt", {"lat": f"{lat:.7f}", "lon": f"{lon:.7f}"})
        ET.SubElement(pt, "ele").text = f"{alt:.2f}"
        if spd is not None:
            spd_ms = spd / 3.6
            ET.SubElement(pt, "speed").text = f"{spd_ms:.2f}"

    indent_xml(root, space="  ")
    xml_str = ET.tostring(root, encoding="unicode")
    with open(gpx_filename, "w", encoding="utf-8") as f:
        f.write(xml_str)
    return True

def write_kml(motec_log, kml_filename, data_log):

    points = list(_iter_gps_trackpoints(data_log))
    if not points:
        return False

    coords_str_list = [f"{lon:.7f},{lat:.7f},{alt:.2f}" for lat, lon, alt, _spd in points]

    root = ET.Element("kml", {"xmlns": "http://www.opengis.net/kml/2.2"})
    doc = ET.SubElement(root, "Document")
    ET.SubElement(doc, "name").text = motec_log.venue_name or "GPS Track Overlay"
    pm = ET.SubElement(doc, "Placemark")
    ET.SubElement(pm, "name").text = "Track Path"
    style = ET.SubElement(pm, "Style")
    lstyle = ET.SubElement(style, "LineStyle")
    ET.SubElement(lstyle, "color").text = "ff0000ff"
    ET.SubElement(lstyle, "width").text = "4"
    ls = ET.SubElement(pm, "LineString")
    ET.SubElement(ls, "extrude").text = "1"
    ET.SubElement(ls, "tessellate").text = "1"
    ET.SubElement(ls, "coordinates").text = "\n".join(coords_str_list)

    indent_xml(root, space="  ")
    xml_str = ET.tostring(root, encoding="unicode")
    with open(kml_filename, "w", encoding="utf-8") as f:
        f.write(xml_str)
    return True
