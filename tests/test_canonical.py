"""Canonical package/CLI entry points for track_telemetry_converter."""
import shutil
import subprocess
import sys


def test_canonical_package_import():
    import track_telemetry_converter
    from track_telemetry_converter.log import DataLog

    assert track_telemetry_converter.__version__
    assert DataLog().channels == {}


def test_canonical_module_entrypoint():
    result = subprocess.run(
        [sys.executable, "-m", "track_telemetry_converter", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_canonical_cli_command():
    executable = shutil.which("track-telemetry")
    if executable is None:
        import pytest

        pytest.skip("track-telemetry console script is not installed")
    result = subprocess.run([executable, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
