from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from track_telemetry_converter import video_sync
from track_telemetry_converter.video_sync import VideoAlignment, VideoSyncError


def _channels(*, bad_speed: bool = False):
    running_start = 829.0
    ld_times = np.arange(0.0, 20.0, 0.04)
    source_times = np.arange(800.0, 860.0, 0.1)

    def values(times, name, *, bad=False):
        if name == "Ground Speed":
            result = 110.0 + (times - running_start) * 0.75
            return result + (10.0 if bad else 0.0)
        if name == "Engine RPM":
            return 5000.0 + (times - running_start) * 30.0
        if name == "GPS Latitude":
            return 37.0 + (times - running_start) * 0.00001
        return -121.0 - (times - running_start) * 0.00001

    names = ("Ground Speed", "Engine RPM", "GPS Latitude", "GPS Longitude")
    ld = {
        name: SimpleNamespace(
            times=ld_times,
            values=values(running_start + ld_times, name, bad=bad_speed and name == "Ground Speed"),
        )
        for name in names
    }
    source = {
        name: SimpleNamespace(times=source_times, values=values(source_times, name))
        for name in names
    }
    return running_start, ld, source


def _fixture(tmp_path, monkeypatch, *, bad_speed=False):
    ld_path = tmp_path / "session.ld"
    rcz_path = tmp_path / "session.rcz"
    video_path = tmp_path / "segment.mp4"
    ldx_path = tmp_path / "session.ldx"
    for path in (ld_path, rcz_path, video_path):
        path.write_bytes(path.suffix.encode("ascii"))
    original_xml = (
        b'<LDXFile><!--keep XML comment--><Layers><Details><String Id="Driver" Value="Keep"/>'
        b'</Details><Layer><Laps><Lap Id="4" Time="123"/></Laps></Layer></Layers><Videos>'
        b'<View Name="Camera View 1"><Source FileName="other.mp4"><SyncPoints Balance="2">'
        b'<SyncPoint DataTime="12" VideoTime="0" Manual="1"/></SyncPoints></Source></View>'
        b"</Videos><Comments><Comment>keep me</Comment></Comments></LDXFile>"
    )
    ldx_path.write_bytes(original_xml)

    running_start, ld_channels, source_channels = _channels(bad_speed=bad_speed)
    monkeypatch.setattr(video_sync, "_load_ld", lambda _: (running_start, ld_channels))
    monkeypatch.setattr(
        video_sync,
        "_load_rcz",
        lambda *_: SimpleNamespace(
            channels=source_channels,
            time_origin_epoch_ms=1_234_000,
        ),
    )
    alignment = VideoAlignment(
        rcz_path=rcz_path,
        video_path=video_path,
        stint=4,
        offset=32.019882917404175,
        video_start=339.0601170825958,
        video_start_utc=1266.0198829174042,
    )
    return {
        "ld": ld_path,
        "rcz": rcz_path,
        "video": video_path,
        "ldx": ldx_path,
        "original_xml": original_xml,
        "alignment": alignment,
    }


def test_source_verification_allows_padded_auxiliary_channel_boundaries():
    start, ld, source = _channels()
    rpm = source["Engine RPM"]
    inside = (rpm.times >= start + 2.0) & (rpm.times <= start + 15.0)
    source["Engine RPM"] = SimpleNamespace(times=rpm.times[inside], values=rpm.values[inside])
    assert video_sync.verify_ld_source(start, ld, source)["Engine RPM"] < 1e-6
    speed = source["Ground Speed"]
    inside = (speed.times >= start + 2.0) & (speed.times <= start + 15.0)
    source["Ground Speed"] = SimpleNamespace(times=speed.times[inside], values=speed.values[inside])
    with pytest.raises(VideoSyncError, match="Ground Speed time range"):
        video_sync.verify_ld_source(start, ld, source)


def test_exports_sync_point_and_preserves_unrelated_ldx_content(tmp_path, monkeypatch):
    files = _fixture(tmp_path, monkeypatch)

    report = video_sync.export_motec_sync(
        files["ld"], files["alignment"], view_name="Camera View 2",
    )

    assert report["data_time_us"] == pytest.approx(-457_920_000.0)
    assert report["changed"] is True
    assert report["backup"] == str(Path(str(files["ldx"]) + ".bak"))
    assert Path(report["backup"]).read_bytes() == files["original_xml"]
    root = ET.parse(files["ldx"]).getroot()
    assert root.findtext("Comments/Comment") == "keep me"
    detail = root.find("Layers/Details/String")
    lap = root.find("Layers/Layer/Laps/Lap")
    assert detail is not None
    assert lap is not None
    assert detail.get("Value") == "Keep"
    assert lap.get("Id") == "4"
    first_source = root.find("Videos/View[@Name='Camera View 1']/Source")
    assert first_source is not None
    assert first_source.get("FileName") == "other.mp4"
    camera = root.find("Videos/View[@Name='Camera View 2']/Source")
    assert camera is not None
    assert camera.get("FileName") == str(files["video"].absolute())
    sync = camera.find("SyncPoints")
    assert sync is not None
    point = sync.find("SyncPoint")
    assert point is not None
    assert sync.get("Balance") == "2"
    assert float(point.get("DataTime", "nan")) == pytest.approx(-457_920_000.0)
    assert point.attrib == {"DataTime": "-457920000", "VideoTime": "0", "Manual": "1"}
    assert set(report["sample_errors"]) >= {"Ground Speed", "Engine RPM"}
    assert "<!--keep XML comment-->" in files["ldx"].read_text(encoding="utf-8")


def test_exact_repeat_is_noop_without_another_backup(tmp_path, monkeypatch):
    files = _fixture(tmp_path, monkeypatch)
    first = video_sync.export_motec_sync(
        files["ld"], files["alignment"], view_name="Camera View 2",
    )
    written = files["ldx"].read_bytes()

    second = video_sync.export_motec_sync(files["ld"], files["alignment"])

    assert first["changed"] is True
    assert second["changed"] is False
    assert second["backup"] is None
    assert files["ldx"].read_bytes() == written
    assert list(tmp_path.glob("session.ldx.bak*")) == [tmp_path / "session.ldx.bak"]


def test_dry_run_validates_without_writing_or_creating_backup(tmp_path, monkeypatch):
    files = _fixture(tmp_path, monkeypatch)

    report = video_sync.export_motec_sync(
        files["ld"], files["alignment"], view_name="Camera View 2", dry_run=True,
    )

    assert report["changed"] is True
    assert report["dry_run"] is True
    assert report["backup"] is None
    assert files["ldx"].read_bytes() == files["original_xml"]
    assert not list(tmp_path.glob("*.bak*"))


def test_copy_to_new_output_even_when_source_ldx_already_has_same_sync(tmp_path, monkeypatch):
    files = _fixture(tmp_path, monkeypatch)
    video_sync.export_motec_sync(
        files["ld"], files["alignment"], view_name="Camera View 2",
    )
    source_bytes = files["ldx"].read_bytes()
    output = tmp_path / "copy" / "session.ldx"

    report = video_sync.export_motec_sync(
        files["ld"],
        files["alignment"],
        output_path=output,
    )

    assert report["changed"] is True
    assert output.is_file()
    assert files["ldx"].read_bytes() == source_bytes
    binding = ET.parse(output).find("Videos/View[@Name='Camera View 1']/Source")
    assert binding is not None
    assert binding.get("FileName") == str((tmp_path / "other.mp4").absolute())


def test_rejects_ld_that_does_not_match_selected_rcz(tmp_path, monkeypatch):
    files = _fixture(tmp_path, monkeypatch, bad_speed=True)

    with pytest.raises(VideoSyncError, match="Ground Speed max sample error"):
        video_sync.export_motec_sync(files["ld"], files["alignment"])

    assert files["ldx"].read_bytes() == files["original_xml"]


def test_rejects_non_unit_rate_and_stale_absolute_video_time(tmp_path, monkeypatch):
    files = _fixture(tmp_path, monkeypatch)
    with pytest.raises(VideoSyncError, match="requires rate 1.0"):
        video_sync.export_motec_sync(
            files["ld"], replace(files["alignment"], rate=1.01),
        )

    stale = replace(files["alignment"], video_start_utc=1267.0)
    with pytest.raises(VideoSyncError, match="does not match"):
        video_sync.export_motec_sync(files["ld"], stale)
    assert files["ldx"].read_bytes() == files["original_xml"]


def test_rejects_non_finite_mapped_data_time_before_reading_or_writing_ldx(
    tmp_path, monkeypatch,
):
    files = _fixture(tmp_path, monkeypatch)
    overflow = replace(files["alignment"], offset=1e308, video_start_utc=None)

    with pytest.raises(VideoSyncError, match="DataTime is non-finite"):
        video_sync.export_motec_sync(files["ld"], overflow)

    assert files["ldx"].read_bytes() == files["original_xml"]


def test_rejects_existing_output_when_it_is_not_the_selected_source_ldx(
    tmp_path, monkeypatch,
):
    files = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "other.ldx"
    output.write_bytes(b"<LDXFile><!--preserve output--></LDXFile>")

    with pytest.raises(VideoSyncError, match="already exists separately"):
        video_sync.export_motec_sync(
            files["ld"],
            files["alignment"],
            output_path=output,
        )

    assert output.read_bytes() == b"<LDXFile><!--preserve output--></LDXFile>"
    assert files["ldx"].read_bytes() == files["original_xml"]


def test_rejects_conflicting_named_view(tmp_path, monkeypatch):
    files = _fixture(tmp_path, monkeypatch)

    with pytest.raises(VideoSyncError, match="already points to another video"):
        video_sync.export_motec_sync(files["ld"], files["alignment"])

    assert files["ldx"].read_bytes() == files["original_xml"]


def test_cli_uses_only_explicit_alignment_arguments(tmp_path, monkeypatch, capsys):
    from track_telemetry_converter.cli import main

    files = _fixture(tmp_path, monkeypatch)
    result = main(
        [
            "motec-sync",
            "--ld",
            str(files["ld"]),
            "--rcz",
            str(files["rcz"]),
            "--video",
            str(files["video"]),
            "--offset",
            str(files["alignment"].offset),
            "--stint",
            "4",
            "--video-start",
            str(files["alignment"].video_start),
            "--video-start-utc",
            str(files["alignment"].video_start_utc),
            "--view-name",
            "Camera View 2",
            "--dry-run",
        ],
    )

    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["data_time_us"] == pytest.approx(-457_920_000.0)
    assert report["dry_run"] is True
    assert files["ldx"].read_bytes() == files["original_xml"]


def test_cli_keeps_loader_diagnostics_off_json_stdout(tmp_path, monkeypatch, capsys):
    from track_telemetry_converter.cli import main

    files = _fixture(tmp_path, monkeypatch)
    load_rcz = video_sync._load_rcz

    def noisy_load_rcz(*args):
        print("RCZ parser diagnostic")
        return load_rcz(*args)

    monkeypatch.setattr(video_sync, "_load_rcz", noisy_load_rcz)
    result = main(
        [
            "motec-sync",
            "--ld",
            str(files["ld"]),
            "--rcz",
            str(files["rcz"]),
            "--video",
            str(files["video"]),
            "--offset",
            str(files["alignment"].offset),
            "--stint",
            "4",
            "--video-start",
            str(files["alignment"].video_start),
            "--video-start-utc",
            str(files["alignment"].video_start_utc),
            "--view-name",
            "Camera View 2",
            "--dry-run",
        ],
    )

    captured = capsys.readouterr()
    assert result == 0
    assert json.loads(captured.out)["data_time_us"] == pytest.approx(-457_920_000.0)
    assert "RCZ parser diagnostic" in captured.err


def test_rejects_running_time_that_drifts_from_declared_native_rate(monkeypatch):
    from track_telemetry_converter._vendor.ldparser import ldData

    drifted = SimpleNamespace(
        name="Running Time",
        data=829.0 + np.arange(20, dtype=np.float64) * 0.002,
        freq=1000.0,
    )
    monkeypatch.setattr(
        ldData,
        "fromfile",
        staticmethod(lambda _: SimpleNamespace(channs=[drifted])),
    )

    with pytest.raises(VideoSyncError, match="regular native-rate time axis"):
        video_sync._load_ld(Path("drifted.ld"))
