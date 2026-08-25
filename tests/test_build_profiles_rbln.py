# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN build profile: detection, runtime lookup, requirements."""

# Standard
from pathlib import Path
import importlib.util

# Third Party
import pytest

# First Party
from setup_extensions.build_profiles import rbln


def _fake_find_spec(installed: set[str]):
    """Return a ``find_spec`` stand-in that only knows the given modules."""

    def find_spec(name: str, package: object = None) -> object:
        return object() if name in installed else None

    return find_spec


@pytest.fixture
def rebel_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Lay out a fake rebel-compiler wheel and point the profile at it."""
    include = tmp_path / "rebel" / "include"
    header = include / rbln.RUNTIME_API_HEADER
    header.parent.mkdir(parents=True)
    header.write_text("// stub\n")
    lib = tmp_path / "tvm"
    lib.mkdir()
    (lib / rbln.RUNTIME_LIBRARY).write_bytes(b"")
    monkeypatch.setattr(rbln, "_site_packages", lambda: tmp_path)
    monkeypatch.delenv("RBLN_RUNTIME_INCLUDE", raising=False)
    monkeypatch.delenv("RBLN_RUNTIME_LIB_DIR", raising=False)
    monkeypatch.delenv("BUILD_WITH_RBLN", raising=False)
    return tmp_path


def test_profile_identity() -> None:
    """The profile is selected via ``BUILD_WITH_RBLN`` and named ``rbln``."""
    profile = rbln.RblnProfile()

    assert profile.name == "rbln"
    assert profile.env_var == "BUILD_WITH_RBLN"


def test_detect_requires_torch_rbln(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection follows the installed ``torch_rbln`` distribution."""
    profile = rbln.RblnProfile()

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(set()))
    assert profile.detect() is False

    monkeypatch.setattr(
        importlib.util, "find_spec", _fake_find_spec({rbln.TORCH_RBLN_MODULE})
    )
    assert profile.detect() is True


def test_detect_tolerates_broken_parent_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``find_spec`` raising (e.g. half-installed package) reads as absent."""

    def raising_find_spec(name: str, package: object = None) -> object:
        raise ValueError("%s.__spec__ is None" % name)

    monkeypatch.setattr(importlib.util, "find_spec", raising_find_spec)

    assert rbln.RblnProfile().detect() is False


def test_runtime_lookup_prefers_env_overrides(
    rebel_runtime: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``RBLN_RUNTIME_INCLUDE`` / ``RBLN_RUNTIME_LIB_DIR`` beat the wheel."""
    override = tmp_path / "checkout"
    (override / rbln.RUNTIME_API_HEADER).parent.mkdir(parents=True)
    (override / rbln.RUNTIME_API_HEADER).write_text("// stub\n")
    (override / rbln.RUNTIME_LIBRARY).write_bytes(b"")
    monkeypatch.setenv("RBLN_RUNTIME_INCLUDE", str(override))
    monkeypatch.setenv("RBLN_RUNTIME_LIB_DIR", str(override))

    assert rbln.rebel_include_dir() == override
    assert rbln.rebel_library_dir() == override


def test_runtime_lookup_reports_missing_pieces(rebel_runtime: Path) -> None:
    """A wheel without headers or without the library reads as absent."""
    (rebel_runtime / "rebel" / "include" / rbln.RUNTIME_API_HEADER).unlink()
    assert rbln.rebel_include_dir() is None
    assert rbln.rebel_library_dir() == rebel_runtime / "tvm"

    (rebel_runtime / "tvm" / rbln.RUNTIME_LIBRARY).unlink()
    assert rbln.rebel_library_dir() is None


def test_build_declares_rbln_ops_extension(rebel_runtime: Path) -> None:
    """With the runtime present the profile emits ``lmcache.rbln_ops``."""
    pytest.importorskip("torch")

    ext_modules, cmdclass = rbln.RblnProfile().build()

    assert [ext.name for ext in ext_modules] == ["lmcache.rbln_ops"]
    ext = ext_modules[0]
    assert "csrc/rbln/kv_transfer.cpp" in ext.sources
    assert str(rebel_runtime / "rebel" / "include") in ext.include_dirs
    assert str(rebel_runtime / "tvm") in ext.library_dirs
    assert "rbln" in ext.libraries
    assert "build_ext" in cmdclass


def test_build_skips_when_runtime_missing_and_autodetected(
    rebel_runtime: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Auto-detected hosts without rebel-compiler fall back to torch."""
    (rebel_runtime / "tvm" / rbln.RUNTIME_LIBRARY).unlink()

    ext_modules, cmdclass = rbln.RblnProfile().build()

    assert ext_modules == []
    assert cmdclass == {}
    assert "rebel-compiler" in capsys.readouterr().err


def test_build_fails_when_runtime_missing_and_explicit(
    rebel_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``BUILD_WITH_RBLN=1`` without the runtime is a hard error."""
    (rebel_runtime / "rebel" / "include" / rbln.RUNTIME_API_HEADER).unlink()
    monkeypatch.setenv("BUILD_WITH_RBLN", "1")

    with pytest.raises(RuntimeError, match="rebel runtime headers"):
        rbln.RblnProfile().build()


def test_requirements_file_lists_torch_rbln_only() -> None:
    """The core requirements pull in torch-rbln but not the private runtime."""
    profile = rbln.RblnProfile()
    req_file = profile.requirements_file()
    assert req_file == "rbln_core.txt"

    lines = [
        line.strip()
        for line in (rbln.ROOT_DIR / "requirements" / req_file).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines == ["torch-rbln"]
    assert profile.extras_requirements() == {}
