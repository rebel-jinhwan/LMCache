# SPDX-License-Identifier: Apache-2.0
"""Rebellions RBLN backend profile: builds ``lmcache.rbln_ops`` against torch-rbln."""

# Standard
from typing import TYPE_CHECKING, Optional
import importlib.util

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.build_profiles import BuildProfile


class RblnProfile(BuildProfile):
    """RBLN extension build profile (requires an installed torch-rbln)."""

    name = "rbln"
    env_var = "BUILD_WITH_RBLN"

    def detect(self) -> bool:
        """Detect RBLN by the presence of the torch-rbln package."""
        return importlib.util.find_spec("torch_rbln") is not None

    def build(self) -> tuple[list["Extension"], dict]:
        """Build ``lmcache.rbln_ops``.

        Only ATen is used, so nothing links against torch-rbln: it supplies the RBLN
        implementations of the copies this extension issues, at runtime.
        """
        # Third Party
        from torch.utils import cpp_extension
        import torch

        abi = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
        ext_modules = [
            cpp_extension.CppExtension(
                "lmcache.rbln_ops",
                sources=["csrc/rbln/pybind_rbln.cpp", "csrc/rbln/kv_transfer.cpp"],
                extra_compile_args={
                    "cxx": ["-std=c++17", "-O3", f"-D_GLIBCXX_USE_CXX11_ABI={abi}"]
                },
            ),
        ]
        return ext_modules, {"build_ext": cpp_extension.BuildExtension}

    def requirements_file(self) -> Optional[str]:
        """No extra requirements beyond torch-rbln itself."""
        return None
