"""Tests for the manual corpus regression gate (tools/corpus_check.py)."""

import importlib.util
import os
import shutil
import tempfile

from conftest import EXAMPLES


def _load_corpus_check():
    path = os.path.join(
        os.path.dirname(__file__), "..", "tools", "corpus_check.py",
    )
    spec = importlib.util.spec_from_file_location("corpus_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_fixtures(tmp_dir):
    for name in ("pbbuddy_sample.csv", "ibt_sample.ibt"):
        shutil.copy(os.path.join(EXAMPLES, name), tmp_dir)


def test_corpus_bless_check_and_drift():
    corpus_check = _load_corpus_check()
    with tempfile.TemporaryDirectory() as tmp_dir:
        _stage_fixtures(tmp_dir)
        assert corpus_check.main(["--dir", tmp_dir, "--bless"]) == 0
        assert corpus_check.main(["--dir", tmp_dir]) == 0
        target = os.path.join(tmp_dir, "pbbuddy_sample.csv")
        with open(target, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines[: len(lines) // 2]) + "\n")
        assert corpus_check.main(["--dir", tmp_dir]) == 1


def test_corpus_missing_dir():
    corpus_check = _load_corpus_check()
    assert corpus_check.main(["--dir", "does-not-exist"]) == 2
