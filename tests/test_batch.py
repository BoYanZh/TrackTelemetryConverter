"""Tests for batch directory mode (smart per-file format detection)."""

import os
import shutil
import tempfile

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
