"""Tests for batch directory mode (smart per-file format detection)."""

import os
import shutil
import tempfile
import zipfile

import pytest
from conftest import EXAMPLES, _run_cli_in_process, _write_backup_rcz, _write_minimal_rcz


def _copy_fixture(name, dest):
    shutil.copy(os.path.join(EXAMPLES, name), dest)


def test_batch_auto_converts_mixed_formats():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _copy_fixture("racechrono_sample.csv", os.path.join(tmp_dir, "a.csv"))
        _copy_fixture("ibt_sample.ibt", os.path.join(tmp_dir, "b.ibt"))
        _write_minimal_rcz(os.path.join(tmp_dir, "c.rcz"))

        result = _run_cli_in_process([tmp_dir, "AUTO"])
        assert result.returncode == 0, result.stdout + result.stderr
        assert "3 converted, 0 skipped, 0 failed" in result.stdout
        for stem in ("a", "b", "c"):
            assert os.path.isfile(os.path.join(tmp_dir, stem + ".ld"))
            assert os.path.isfile(os.path.join(tmp_dir, stem + ".ldx"))


def test_batch_second_run_skips_existing_without_force():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _copy_fixture("racechrono_sample.csv", os.path.join(tmp_dir, "a.csv"))

        first = _run_cli_in_process([tmp_dir, "AUTO"])
        assert first.returncode == 0, first.stdout + first.stderr

        second = _run_cli_in_process([tmp_dir, "AUTO"])
        assert second.returncode == 0, second.stdout + second.stderr
        assert "0 converted, 1 skipped, 0 failed" in second.stdout
        assert "already exists" in second.stdout


def test_batch_recursive_and_explicit_type_filter():
    with tempfile.TemporaryDirectory() as tmp_dir:
        sub_dir = os.path.join(tmp_dir, "sub")
        os.mkdir(sub_dir)
        _copy_fixture("racechrono_sample.csv", os.path.join(tmp_dir, "a.csv"))
        _copy_fixture("ibt_sample.ibt", os.path.join(sub_dir, "b.ibt"))

        flat = _run_cli_in_process([tmp_dir, "AUTO"])
        assert flat.returncode == 0, flat.stdout + flat.stderr
        assert "1 converted" in flat.stdout
        assert not os.path.exists(os.path.join(sub_dir, "b.ld"))

        filtered = _run_cli_in_process([tmp_dir, "IBT", "--recursive"])
        assert filtered.returncode == 0, filtered.stdout + filtered.stderr
        assert "1 converted" in filtered.stdout
        assert os.path.isfile(os.path.join(sub_dir, "b.ld"))


def test_batch_skips_backup_archive_without_session_all():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _write_backup_rcz(os.path.join(tmp_dir, "backup.rcz"))
        _copy_fixture("racechrono_sample.csv", os.path.join(tmp_dir, "a.csv"))

        result = _run_cli_in_process([tmp_dir, "AUTO"])
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 converted, 1 skipped, 0 failed" in result.stdout
        assert "backup archive" in result.stdout
        assert os.path.isfile(os.path.join(tmp_dir, "a.ld"))


def test_batch_session_all_expands_backup_archive():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _write_backup_rcz(os.path.join(tmp_dir, "backup.rcz"))

        result = _run_cli_in_process([tmp_dir, "AUTO", "--session", "all"])
        assert result.returncode == 0, result.stdout + result.stderr
        sessions_dir = os.path.join(tmp_dir, "backup_sessions")
        assert os.path.isfile(os.path.join(sessions_dir, "session_20260101_1000.ld"))
        assert os.path.isfile(os.path.join(sessions_dir, "session_20260102_1100_stint0.ld"))
        assert os.path.isfile(os.path.join(sessions_dir, "session_20260102_1100_stint1.ld"))


def test_batch_rejects_single_file_options():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _copy_fixture("racechrono_sample.csv", os.path.join(tmp_dir, "a.csv"))

        for extra in (["--output", "out.ld"], ["--session", "session_1"],
                      ["--stint", "0"], ["--lap", "1"], ["--list-sessions"]):
            result = _run_cli_in_process([tmp_dir, "AUTO"] + extra)
            assert result.returncode != 0, extra
            assert "ERROR" in result.stdout


def test_batch_empty_directory_errors():
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = _run_cli_in_process([tmp_dir, "AUTO"])
        assert result.returncode != 0
        assert "No convertible logs" in result.stdout


def test_auto_whitelist_matches_shared_constant():
    from track_telemetry_converter.cli import _BATCH_AUTO_EXTENSIONS
    from track_telemetry_converter.parsers import SUPPORTED_INPUT_EXTENSIONS

    assert set(_BATCH_AUTO_EXTENSIONS) == set(SUPPORTED_INPUT_EXTENSIONS)
    # Spot-check: convertible formats in, video/outputs/sidecars out.
    for ext in (".rcz", ".csv", ".vbo", ".dlap", ".ibt",
                ".xrk", ".xrz", ".fit", ".log", ".can", ".trc", ".asc"):
        assert ext in SUPPORTED_INPUT_EXTENSIONS
    for ext in (".mp4", ".mkv", ".webm", ".ld", ".ldx", ".dbc",
                ".yaml", ".txt", ".png", ""):
        assert ext not in SUPPORTED_INPUT_EXTENSIONS


def test_batch_auto_skips_video_and_unrelated_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _copy_fixture("racechrono_sample.csv", os.path.join(tmp_dir, "a.csv"))
        for name, content in (("clip.mp4", b"\x00\x00\x00 ftyp"),
                              ("clip.mkv", b"\x1a\x45\xdf\xa3"),
                              ("notes.txt", b"hello"),
                              ("job.yaml", b"key: value")):
            with open(os.path.join(tmp_dir, name), "wb") as handle:
                handle.write(content)

        result = _run_cli_in_process([tmp_dir, "AUTO"])
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 converted, 4 skipped, 0 failed" in result.stdout
        assert "unsupported extension" in result.stdout
        assert os.path.isfile(os.path.join(tmp_dir, "a.ld"))


def test_batch_help_lists_whitelisted_extensions():
    from track_telemetry_converter.cli import build_parser

    help_text = build_parser().format_help()
    assert ".rcz" in help_text
    assert ".mp4" not in help_text


def _write_root_multistint_rcz(path):
    """Flatten the two-stint second session from the nested-backup fixture."""
    backup = path.with_name("nested_backup_fixture.rcz")
    _write_backup_rcz(backup)
    session_prefix = "sessions/session_20260102_1100/"
    with zipfile.ZipFile(backup, "r") as source:
        entries = [
            (item.filename[len(session_prefix):], source.read(item.filename))
            for item in source.infolist()
            if item.filename.startswith(session_prefix)
        ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in entries:
            archive.writestr(name, contents)
    backup.unlink()


@pytest.mark.parametrize("jobs", [1, 2])
def test_batch_backup_session_all_second_run_is_skipped(tmp_path, jobs):
    _write_backup_rcz(tmp_path / "backup.rcz")

    first = _run_cli_in_process(
        [str(tmp_path), "AUTO", "--session", "all", "--jobs", str(jobs)],
    )
    assert first.returncode == 0, first.stdout + first.stderr
    assert "1 converted, 0 skipped, 0 failed" in first.stdout

    second = _run_cli_in_process(
        [str(tmp_path), "AUTO", "--session", "all", "--jobs", str(jobs)],
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert "0 converted, 1 skipped, 0 failed" in second.stdout


@pytest.mark.parametrize("jobs", [1, 2])
def test_batch_root_multistint_second_run_is_skipped(tmp_path, jobs):
    source = tmp_path / "multi_stint.rcz"
    _write_root_multistint_rcz(source)

    first = _run_cli_in_process([str(tmp_path), "AUTO", "--jobs", str(jobs)])
    assert first.returncode == 0, first.stdout + first.stderr
    assert "1 converted, 0 skipped, 0 failed" in first.stdout
    assert (tmp_path / "multi_stint_stint0.ld").is_file()
    assert (tmp_path / "multi_stint_stint1.ld").is_file()

    second = _run_cli_in_process([str(tmp_path), "AUTO", "--jobs", str(jobs)])
    assert second.returncode == 0, second.stdout + second.stderr
    assert "0 converted, 1 skipped, 0 failed" in second.stdout


def test_batch_parallel_vbo_custom_channel_spec_is_picklable(tmp_path):
    for stem in ("first", "second"):
        _copy_fixture("racechrono_sample.csv", tmp_path / f"{stem}.csv")

    result = _run_cli_in_process(
        [
            str(tmp_path), "AUTO", "--jobs", "2", "--vbo",
            "--vbo-columns", "MyRPM=Engine RPM",
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 converted, 0 skipped, 0 failed" in result.stdout
    for stem in ("first", "second"):
        vbo = tmp_path / f"{stem}.vbo"
        assert vbo.is_file()
        assert "MyRPM" in vbo.read_text(encoding="utf-8")
