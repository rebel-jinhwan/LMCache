# SPDX-License-Identifier: Apache-2.0
"""Engine-side prerequisites of RBLN LMCache-driven transfer, for CI only.

``vllm_e2e.py --transfer-mode lmcache_driven`` puts this directory on the vLLM
process's ``PYTHONPATH``, so every interpreter vLLM starts (API server, engine
core) applies two vLLM-RBLN changes that are not upstream yet:

- KV buffers are allocated with ``torch.rbln.ipc.empty_exportable``, the only
  memory ``torch.rbln.ipc.export_tensor`` accepts.
- ``LMCacheMPConnector`` joins the connectors the dynamic-KV resize is open
  for, so vLLM-RBLN registers the KV cache after ``materialize()`` placed it.
  Registering earlier exports unplaced memory, which the runtime pins flat on
  one chiplet.

Remove this shim once vLLM-RBLN (or the runtime) provides both.
"""

# Standard
from collections.abc import Callable, Sequence
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
import sys

# Third Party
import torch


def _patch_runner(module: ModuleType) -> None:
    runner_cls = module.RBLNModelRunner
    allocate = runner_cls._allocate_kv_cache_tensors

    def allocate_exportable(self: object, kv_cache_config: object) -> object:
        empty = torch.empty

        def exportable_empty(*args: object, **kwargs: object) -> torch.Tensor:
            device = kwargs.get("device")
            if (
                len(args) == 1
                and kwargs.get("dtype") is torch.int8
                and device is not None
                and torch.device(device).type == "rbln"  # type: ignore[arg-type]
            ):
                return torch.rbln.ipc.empty_exportable(  # type: ignore[attr-defined]
                    (args[0],),
                    torch.int8,
                    device=torch.device(device),  # type: ignore[arg-type]
                )
            return empty(*args, **kwargs)  # type: ignore[arg-type]

        torch.empty = exportable_empty  # type: ignore[assignment]
        try:
            return allocate(self, kv_cache_config)
        finally:
            torch.empty = empty

    runner_cls._allocate_kv_cache_tensors = allocate_exportable


def _patch_utils(module: ModuleType) -> None:
    module.DYNAMIC_KV_SUPPORTED_CONNECTORS = (
        *module.DYNAMIC_KV_SUPPORTED_CONNECTORS,
        "LMCacheMPConnector",
    )


_PATCHES: dict[str, Callable[[ModuleType], None]] = {
    "vllm_rbln.v1.worker.rbln_model_runner": _patch_runner,
    "vllm_rbln.v1.worker.utils": _patch_utils,
}


class _PatchingLoader(Loader):
    def __init__(self, loader: Loader, patch: Callable[[ModuleType], None]) -> None:
        self._loader = loader
        self._patch = patch

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        self._patch(module)


class _PatchingFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        patch = _PATCHES.get(fullname)
        if patch is None:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PatchingLoader(spec.loader, patch)
                return spec
        return None


sys.meta_path.insert(0, _PatchingFinder())
