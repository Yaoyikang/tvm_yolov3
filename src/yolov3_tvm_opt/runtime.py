from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from .tvm_import import tvm_import


@dataclass(frozen=True)
class Artifact:
    lib_path: str
    meta: Dict[str, Any]


def load_artifact(artifact_dir: str) -> Artifact:
    lib_path = os.path.join(artifact_dir, "model.so")
    meta_path = os.path.join(artifact_dir, "meta.json")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return Artifact(lib_path=lib_path, meta=meta)


def create_vm(artifact: Artifact, device: str = "cuda", device_id: int = 0, profile: bool = False):
    tvm = tvm_import()
    from tvm import relax
    from tvm.runtime import _tensor as rt

    if device == "cuda":
        dev = tvm.cuda(device_id)
    elif device == "cpu":
        dev = tvm.cpu(device_id)
    else:
        raise ValueError(f"Unknown device: {device}")

    lib = tvm.runtime.load_module(artifact.lib_path)
    ex = relax.vm_build.VMExecutable(lib)
    vm = relax.VirtualMachine(ex, dev, profile=bool(profile))
    return vm, dev, rt


def benchmark(
    vm,
    rt,
    dev,
    input_name: str,
    input_data: np.ndarray,
    warmup: int = 10,
    iters: int = 200,
) -> Tuple[float, Dict[str, np.ndarray]]:
    # returns (avg_ms, outputs)
    # VM entry expects positional args; input_name is kept for CLI compatibility.
    x = rt.tensor(input_data, device=dev)
    fn = vm["main"]

    for _ in range(warmup):
        _ = fn(x)

    # CUDA kernels are async; sync to get accurate wall-clock timing.
    if hasattr(dev, "sync"):
        dev.sync()

    t0 = time.time()
    out = None
    for _ in range(iters):
        out = fn(x)
    if hasattr(dev, "sync"):
        dev.sync()
    t1 = time.time()

    outputs: Dict[str, np.ndarray] = {}
    if out is not None:
        # Many models return a tuple/list. Normalize to dict.
        if hasattr(out, "numpy"):
            outputs["0"] = out.numpy()
        elif isinstance(out, (list, tuple)) or (
            hasattr(out, "__len__") and hasattr(out, "__getitem__")
        ):
            for i in range(len(out)):
                item = out[i]
                outputs[str(i)] = item.numpy() if hasattr(item, "numpy") else np.array(item)
        else:
            outputs["0"] = np.array(out)

    avg_ms = (t1 - t0) * 1000.0 / iters
    return avg_ms, outputs
