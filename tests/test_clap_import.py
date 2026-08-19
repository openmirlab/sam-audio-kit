"""Verify the optional CLAP ranker does not inherit host process CLI arguments."""

import sys

from sam_audio_kit.sam_audio.ranking import clap


def test_clap_data_import_isolates_and_restores_process_argv(monkeypatch):
    sentinel = object()
    observed = []
    original_argv = sys.argv
    requested_argv = ["provider", "--port", "8028", "--device", "cuda:0"]

    def fake_import(module_name):
        observed.append((module_name, sys.argv[:]))
        return sentinel

    monkeypatch.setattr(clap, "import_module", fake_import)
    sys.argv = requested_argv
    try:
        assert clap._load_laion_data_module() is sentinel
        assert sys.argv == requested_argv
    finally:
        sys.argv = original_argv

    assert observed == [("laion_clap.training.data", ["provider"])]
    assert sys.argv is original_argv


def test_clap_data_import_restores_argv_after_import_failure(monkeypatch):
    original_argv = sys.argv
    requested_argv = ["provider", "--port", "8028"]

    def fail_import(module_name):
        raise RuntimeError(f"failed to import {module_name}")

    monkeypatch.setattr(clap, "import_module", fail_import)
    sys.argv = requested_argv
    try:
        try:
            clap._load_laion_data_module()
        except RuntimeError as error:
            assert str(error) == "failed to import laion_clap.training.data"
        else:
            raise AssertionError("expected the fake import to fail")
        assert sys.argv == requested_argv
    finally:
        sys.argv = original_argv

    assert sys.argv is original_argv
