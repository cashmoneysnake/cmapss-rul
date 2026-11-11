"""Smoke tests: the repo layout and download script are wired up correctly."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_download_script():
    spec = importlib.util.spec_from_file_location(
        "download_data", ROOT / "scripts" / "download_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evaluation_protocol_doc_exists():
    assert (ROOT / "docs" / "evaluation_protocol.md").exists()


def test_download_script_expects_all_twelve_files():
    mod = _load_download_script()
    assert len(mod.EXPECTED_FILES) == 12
    assert set(mod.EXPECTED_ENGINES) == {"FD001", "FD002", "FD003", "FD004"}


def test_download_script_has_a_fallback_url():
    mod = _load_download_script()
    assert len(mod.URLS) >= 2
