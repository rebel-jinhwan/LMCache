# SPDX-License-Identifier: Apache-2.0
"""RBLN NPU backend profile.

Builds ``lmcache.rbln_ops`` against the rebel runtime.  Both the headers and
the library it links come from the ``rebel-compiler`` distribution already
installed on an RBLN host -- headers from ``<rebel>/include``, ``librbln.so``
from the ``tvm`` package that ships with it.  Taking both from one installation
means they cannot drift apart; a vendored copy of the headers would fail
silently against a different runtime instead of failing at build time.

Overrides for source-tree builds: ``RBLN_RUNTIME_INCLUDE`` and
``RBLN_RUNTIME_LIB_DIR``.
"""

# Standard
from typing import TYPE_CHECKING, Optional
import os
import sysconfig

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.build_profiles import BuildProfile

RBLN_SOURCES = [
    "csrc/rbln/pybind_rbln.cpp",
    "csrc/rbln/kv_transfer.cpp",
]


def _site_packages() -> Optional[str]:
    """Directory the rebel / tvm distributions are installed into."""
    try:
        return sysconfig.get_paths()["purelib"]
    except Exception:
        return None


def rebel_include_dir() -> Optional[str]:
    """Directory holding ``rebel/runtime/api/rbln_runtime_api.h``.

    ``RBLN_RUNTIME_INCLUDE`` wins so a build can be pointed at a runtime
    source tree; otherwise the headers shipped by the installed
    ``rebel-compiler`` wheel are used.
    """
    override = os.environ.get("RBLN_RUNTIME_INCLUDE")
    if override:
        return override if os.path.isdir(override) else None
    site = _site_packages()
    if site is None:
        return None
    candidate = os.path.join(site, "rebel", "include")
    return candidate if os.path.isdir(candidate) else None


def rebel_lib_dir() -> Optional[str]:
    """Directory holding ``librbln.so``.

    ``RBLN_RUNTIME_LIB_DIR`` wins; otherwise the ``tvm`` package that
    rebel-compiler installs alongside the headers.
    """
    candidates = [os.environ.get("RBLN_RUNTIME_LIB_DIR")]
    site = _site_packages()
    if site is not None:
        candidates.append(os.path.join(site, "tvm"))
    for candidate in candidates:
        if candidate and os.path.exists(os.path.join(candidate, "librbln.so")):
            return candidate
    return None


class RblnProfile(BuildProfile):
    """RBLN NPU extension build profile."""

    name = "rbln"
    env_var = "BUILD_WITH_RBLN"

    def detect(self) -> bool:
        """Detect RBLN by the rebel runtime headers *and* library both being
        resolvable.  Either one alone cannot produce a linked extension."""
        return rebel_include_dir() is not None and rebel_lib_dir() is not None

    def build(self) -> tuple[list["Extension"], dict]:
        """Build ``lmcache.rbln_ops`` against the installed rebel runtime.

        Raises:
            RuntimeError: If the headers or ``librbln.so`` cannot be located.
                This is the explicit-request path (``BUILD_WITH_RBLN=1``);
                auto-detection never reaches it.
        """
        # Third Party
        from torch.utils import cpp_extension

        print("Building RBLN extensions")
        include_dir = rebel_include_dir()
        if include_dir is None:
            raise RuntimeError(
                "rebel runtime headers not found. Install rebel-compiler "
                "(>= 0.11.1.dev322, the first release to ship rebel/include) "
                "or set RBLN_RUNTIME_INCLUDE."
            )
        lib_dir = rebel_lib_dir()
        if lib_dir is None:
            raise RuntimeError(
                "librbln.so not found. Install rebel-compiler or set "
                "RBLN_RUNTIME_LIB_DIR."
            )

        ext_modules = [
            cpp_extension.CppExtension(
                "lmcache.rbln_ops",
                sources=RBLN_SOURCES,
                include_dirs=["csrc", include_dir],
                library_dirs=[lib_dir],
                libraries=["rbln"],
                runtime_library_dirs=[lib_dir],
                extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
            ),
        ]
        cmdclass = {"build_ext": cpp_extension.BuildExtension}
        return ext_modules, cmdclass

    def requirements_file(self) -> Optional[str]:
        """RBLN adds nothing to ``install_requires``.

        torch-rbln is exposed as the ``rbln`` extra instead -- see
        :meth:`extras_requirements` and ``requirements/rbln.txt`` for why it
        cannot be a core requirement.
        """
        return None

    def extras_requirements(self) -> dict[str, str]:
        """Return the RBLN optional extras.

        Returns:
            Mapping with the ``"rbln"`` extra (``pip install lmcache[rbln]``),
            which pulls torch-rbln -- the package that registers the ``rbln``
            torch backend and therefore the one that makes the device
            detectable at all.
        """
        return {"rbln": "rbln.txt"}
