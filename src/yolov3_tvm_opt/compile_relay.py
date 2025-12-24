from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from .tvm_import import tvm_import


@dataclass(frozen=True)
class CompileMeta:
    onnx_path: str
    input_name: str
    input_shape: Tuple[int, int, int, int]  # NCHW
    input_dtype: str
    target: str
    tuned: bool


def _infer_onnx_input(onnx_model: Any) -> Tuple[str, Tuple[int, int, int, int], str]:
    # Minimal inference: assume first input is the image tensor.
    inp = onnx_model.graph.input[0]
    name = inp.name

    dtype = "float32"

    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) != 4 or any(d == 0 for d in dims):
        dims = [1, 3, 416, 416]
    shape = (int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3]))
    return name, shape, dtype


def compile_relay(
    onnx_path: str,
    out_dir: str,
    target: str,
    input_name: Optional[str] = None,
    input_shape: Optional[Tuple[int, int, int, int]] = None,
    input_dtype: str = "float32",
    tune: bool = False,
    work_dir: Optional[str] = None,
    max_trials_global: int = 2000,
    max_trials_per_task: Optional[int] = None,
    num_trials_per_iter: int = 64,
    use_fp16: bool = False,
    relax_pipeline: str = "default",
) -> CompileMeta:
    """Compile YOLOv3 ONNX.

    Notes on this repo's TVM build:
    - `tvm.relay` may be unavailable.
    - `tvm.contrib.graph_executor` may be unavailable.

    Therefore this function uses **Relax VM** compilation and exports:
    - model.so (Relax VMExecutable)
    - meta.json

    Parameters
    - target: "llvm" or "cuda"
    - tune: reserved (not implemented in this minimal project)
    """

    if (work_dir is not None) and (not tune):
        raise ValueError("--work_dir is only meaningful with --tune")

    tvm = tvm_import()
    import onnx

    from tvm import relax
    from tvm.relax.frontend import detach_params
    from tvm.relax.frontend import onnx as relax_onnx

    os.makedirs(out_dir, exist_ok=True)

    onnx_model = onnx.load(onnx_path)

    inferred_name, inferred_shape, inferred_dtype = _infer_onnx_input(onnx_model)
    if input_name is None:
        input_name = inferred_name
    if input_shape is None:
        input_shape = inferred_shape
    if input_dtype is None:
        input_dtype = inferred_dtype

    # Relax ONNX importer expects shape_dict values as List
    shape_dict: Dict[str, list] = {input_name: list(input_shape)}
    dtype_dict: Dict[str, str] = {input_name: input_dtype}

    mod = relax_onnx.from_onnx(onnx_model, shape_dict=shape_dict, dtype_dict=dtype_dict)
    mod, params = detach_params(mod)

    # detach_params may return Dict[str, List[Tensor]] in this build.
    normalized_params: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, list):
            if len(v) != 1:
                raise ValueError(f"Unexpected param list length for {k}: {len(v)}")
            normalized_params[k] = v[0]
        else:
            normalized_params[k] = v

    tvm_target = tvm.target.Target(target, host="llvm" if target == "cuda" else None)

    # Build an optimization pipeline for Relax IR.
    relax_passes = [relax.get_pipeline(relax_pipeline)]
    # A few extra generic cleanups/fusions that are usually safe.
    relax_passes.extend(
        [
            relax.transform.CanonicalizeBindings(),
            relax.transform.EliminateCommonSubexpr(),
            relax.transform.FoldConstant(),
            relax.transform.DeadCodeElimination(),
            relax.transform.DecomposeOpsForInference(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ]
    )
    if use_fp16:
        # Keep outputs in fp32 by default; allow fp16 inputs.
        relax_passes.insert(1, relax.transform.ToMixedPrecision(out_dtype="float32"))
    relax_pipeline_pass = tvm.ir.transform.Sequential(relax_passes)

    mod = relax_pipeline_pass(mod)

    tuned_effective = False
    with tvm.transform.PassContext(opt_level=3):
        if tune:
            from tvm.meta_schedule import relax_integration as ms_relax

            try:
                db = ms_relax.tune_relax(
                    mod=mod,
                    params=normalized_params,
                    target=tvm_target,
                    work_dir=work_dir or os.path.join(out_dir, "ms_workdir"),
                    max_trials_global=int(max_trials_global),
                    max_trials_per_task=max_trials_per_task,
                    num_trials_per_iter=int(num_trials_per_iter),
                )
                ex = ms_relax.compile_relax(
                    database=db,
                    mod=mod,
                    target=tvm_target,
                    params=normalized_params,
                    enable_warning=False,
                )
                tuned_effective = True
            except ValueError as e:
                if "No tasks to tune" not in str(e):
                    raise
                # Fall back to non-tuned compilation.
                tune = False

        if not tune:
            tir_pipeline = tvm.tir.pipeline.get_default_tir_pipeline(tvm_target)
            if target == "cuda":
                # Ensure lowered CUDA TIR has thread bindings (avoid VerifyMemory failures).
                tir_pipeline = tvm.ir.transform.Sequential(
                    [tvm.tir.transform.DefaultGPUSchedule(), tir_pipeline]
                )
            ex = relax.build(mod, target=tvm_target, params=normalized_params, tir_pipeline=tir_pipeline)

    lib_path = os.path.join(out_dir, "model.so")
    ex.export_library(lib_path)

    meta = CompileMeta(
        onnx_path=os.path.abspath(onnx_path),
        input_name=input_name,
        input_shape=tuple(input_shape),
        input_dtype=input_dtype,
        target=str(tvm_target),
        tuned=bool(tuned_effective),
    )
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    return meta
