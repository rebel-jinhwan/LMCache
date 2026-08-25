# SPDX-License-Identifier: Apache-2.0
"""RBLN (Rebellions NPU) backend profile.

Builds the ``lmcache.rbln_ops`` extension: the MLA block KV transfer that
drives the rebel runtime's async DMA queue directly (see ``csrc/rbln/``).
The extension compiles against the rebel runtime headers and links
``librbln.so``; both ship inside the ``rebel-compiler`` wheel
(``rebel/include`` and ``tvm/librbln.so``), so a single installed wheel keeps
header and library in step.

``torch.rbln`` is contributed by the ``torch-rbln`` package through a torch
backend entry point.  ``torch-rbln`` itself only declares ``torch``,
``scipy``, ``libcst`` and ``PyYAML`` as runtime dependencies; the
``rebel-compiler`` runtime it imports at start-up is *not* pulled in
transitively and is distributed from Rebellions' private package index, so
it cannot be listed in ``install_requires`` (see
``requirements/rbln_core.txt``).

Environment overrides:
    RBLN_RUNTIME_INCLUDE  directory holding ``rebel/runtime/api/*.h``
    RBLN_RUNTIME_LIB_DIR  directory holding ``librbln.so``
"""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import importlib.util
import os
import sys
import sysconfig

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.build_profiles import BuildProfile

TORCH_RBLN_MODULE = "torch_rbln"
REBEL_RUNTIME_MODULE = "rebel"
ROOT_DIR = Path(__file__).resolve().parents[2]
CSRC_DIR = str(ROOT_DIR / "csrc")
RUNTIME_API_HEADER = Path("rebel") / "runtime" / "api" / "rbln_runtime_api.h"
RUNTIME_LIBRARY = "librbln.so"


def _module_installed(name: str) -> bool:
    """Return True when an importable distribution provides module ``name``.

    Uses :func:`importlib.util.find_spec` so the module is located without
    being imported: importing ``torch_rbln`` on a build host would try to
    open the NPU runtime.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _site_packages() -> Path:
    """Return the ``purelib`` directory of the interpreter running the build."""
    return Path(sysconfig.get_paths()["purelib"])


def rebel_include_dir() -> Optional[Path]:
    """Locate the directory holding the rebel runtime public headers.

    ``RBLN_RUNTIME_INCLUDE`` wins when set; otherwise the ``rebel/include``
    tree of the installed ``rebel-compiler`` wheel is used.

    Returns:
        The include directory, or ``None`` when ``rbln_runtime_api.h`` is not
        found beneath the candidate.
    """
    override = os.environ.get("RBLN_RUNTIME_INCLUDE")
    candidate = Path(override) if override else _site_packages() / "rebel" / "include"
    if (candidate / RUNTIME_API_HEADER).is_file():
        return candidate
    return None


def rebel_library_dir() -> Optional[Path]:
    """Locate the directory holding ``librbln.so``.

    ``RBLN_RUNTIME_LIB_DIR`` wins when set; otherwise the ``tvm`` package of
    the installed ``rebel-compiler`` wheel, which carries the runtime library.

    Returns:
        The library directory, or ``None`` when ``librbln.so`` is not there.
    """
    override = os.environ.get("RBLN_RUNTIME_LIB_DIR")
    candidate = Path(override) if override else _site_packages() / "tvm"
    if (candidate / RUNTIME_LIBRARY).is_file():
        return candidate
    return None


class RblnProfile(BuildProfile):
    """RBLN NPU extension build profile."""

    name = "rbln"
    env_var = "BUILD_WITH_RBLN"

    def detect(self) -> bool:
        """Detect RBLN by the presence of the ``torch_rbln`` package.

        There is no RBLN compiler to look for in ``PATH`` and probing
        ``torch.rbln.is_available()`` would open the device, so the
        installed ``torch-rbln`` distribution is the build-time signal.
        """
        return _module_installed(TORCH_RBLN_MODULE)

    def build(self) -> tuple[list["Extension"], dict]:
        """Build ``lmcache.rbln_ops`` against the installed rebel runtime.

        When the rebel headers or ``librbln.so`` cannot be located the
        extension is skipped with a warning if the profile was merely
        auto-detected, so a host with ``torch-rbln`` but no runtime still
        installs LMCache on the torch fallback.

        Raises:
            RuntimeError: If ``BUILD_WITH_RBLN=1`` was requested explicitly
                but the rebel runtime headers or library are missing.
        """
        include_dir = rebel_include_dir()
        lib_dir = rebel_library_dir()
        if include_dir is None or lib_dir is None:
            missing = (
                "rebel runtime headers (%s)" % RUNTIME_API_HEADER
                if include_dir is None
                else RUNTIME_LIBRARY
            )
            message = (
                "%s not found. Install rebel-compiler (>=0.11.1.dev322 ships "
                "rebel/include) from the RBLN package index "
                "(https://pypi.rbln.ai/simple), or point RBLN_RUNTIME_INCLUDE / "
                "RBLN_RUNTIME_LIB_DIR at a runtime checkout." % missing
            )
            if self.is_explicitly_requested():
                raise RuntimeError(message)
            print(
                "warning: %s Skipping lmcache.rbln_ops; RblnDeviceOps will stay "
                "on the torch baseline." % message,
                file=sys.stderr,
            )
            return [], {}

        # Third Party
        from torch.utils import cpp_extension

        print("Building RBLN extensions")
        ext_modules = [
            cpp_extension.CppExtension(
                "lmcache.rbln_ops",
                sources=[
                    "csrc/rbln/kv_transfer.cpp",
                    "csrc/rbln/pybind.cpp",
                ],
                include_dirs=[CSRC_DIR, str(include_dir)],
                library_dirs=[str(lib_dir)],
                libraries=["rbln"],
                runtime_library_dirs=[str(lib_dir)],
                extra_compile_args={"cxx": ["-O2", "-std=c++17"]},
            ),
        ]
        cmdclass = {"build_ext": cpp_extension.BuildExtension}
        return ext_modules, cmdclass

    def requirements_file(self) -> Optional[str]:
        """RBLN core requirements file."""
        return "rbln_core.txt"
