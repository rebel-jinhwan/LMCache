# SPDX-License-Identifier: Apache-2.0
"""Tests for the build profiles' requirements contract.

A profile names its requirements by filename, and ``_read_requirements`` returns
an empty list for a path that does not exist. A renamed or deleted file
therefore ships a wheel with the dependency missing rather than failing the
build, surfacing later as an import error on a user's machine.
"""

# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from setup_extensions.build_profiles import BuildProfile
from setup_extensions.build_profiles.rbln import RblnProfile
from setup_extensions.policy import _discover_platforms

REQUIREMENTS_DIR = Path(__file__).resolve().parents[1] / "requirements"


def _requirement_names(path: Path) -> list[str]:
    """Package names declared in a requirements file, markers stripped."""
    names = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for separator in (";", "=", ">", "<", "!", "["):
            line = line.split(separator, maxsplit=1)[0]
        names.append(line.strip())
    return names


@pytest.fixture(scope="module")
def profiles() -> list[BuildProfile]:
    """Every profile ``BuildPolicy`` would consider, toolchain or not."""
    return _discover_platforms()


def test_every_profile_is_discovered(profiles: list[BuildProfile]) -> None:
    """Discovery is filesystem-based, so a new profile needs no registration."""
    assert {profile.name for profile in profiles} >= {
        "cuda",
        "musa",
        "rbln",
        "rocm",
        "sycl",
    }


def test_named_requirements_files_exist(profiles: list[BuildProfile]) -> None:
    """Both the core file and every extra must resolve to a real file."""
    for profile in profiles:
        core = profile.requirements_file()
        if core is not None:
            assert (REQUIREMENTS_DIR / core).is_file(), (
                "%s.requirements_file() names a missing file: %s" % (profile.name, core)
            )
        for extra_name, extra_file in profile.extras_requirements().items():
            assert (REQUIREMENTS_DIR / extra_file).is_file(), (
                "%s extra %r names a missing file: %s"
                % (profile.name, extra_name, extra_file)
            )


def test_rbln_core_requires_torch_rbln() -> None:
    """Without it ``torch.rbln`` does not exist and every RBLN path is dead,
    which makes it a core requirement rather than an optional one."""
    profile = RblnProfile()
    assert profile.requirements_file() == "rbln_core.txt"
    assert _requirement_names(REQUIREMENTS_DIR / "rbln_core.txt") == ["torch-rbln"]


def test_rbln_core_does_not_pin_torch_itself() -> None:
    """Repeating the pin here would conflict with the unpinned ``torch`` in
    ``requirements/common.txt``, which ``setup.py`` concatenates this onto."""
    assert "torch" not in _requirement_names(REQUIREMENTS_DIR / "rbln_core.txt")
