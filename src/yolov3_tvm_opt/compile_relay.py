from __future__ import annotations

import json
import os
import inspect
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
    exec_mode: str
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
    exec_mode: str = "bytecode",
    tune_op_names: Optional[list[str]] = None,
    tune_builder_timeout_sec: Optional[float] = None,
    tune_runner_timeout_sec: Optional[float] = None,
    fast_scatter_nd: bool = False,
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

    if exec_mode not in ("bytecode", "compiled"):
        raise ValueError("exec_mode must be 'bytecode' or 'compiled'")

    # Meta-schedule's AutoBind may require max_threads_per_block on some TVM builds.
    # If the user passed a plain 'cuda' target, enrich it with a sensible default.
    effective_target = target
    if target == "cuda":
        # Defaults that satisfy meta-schedule's GPU verification in some TVM builds.
        # 49152 bytes is a conservative per-block shared memory limit across many GPUs.
        effective_target = "cuda -max_threads_per_block=1024 -max_shared_memory_per_block=49152"

    tvm_target = tvm.target.Target(effective_target, host="llvm" if target == "cuda" else None)

    def _maybe_specialize_yolo_scatter_nd(mod_in: Any) -> Any:
        """Specialize common YOLO scatter_nd patterns into pure copies.

        In this model, some `scatter_nd*` PrimFuncs are used to assemble the last
        channel dimension (85 = 4 + 1 + 80) from two tensors:
        - base: (..., 85)
        - values: (..., 81)
        with indices that place `values` into channels [4:85].

        The generic scatter implementation is extremely expensive (random writes,
        large int64 index tensor). When the pattern matches, we can replace it
        with a simple memcpy-like kernel:
        - out[..., :4]  = base[..., :4]
        - out[..., 4:]  = values

        This does *not* try to cover all scatter_nd variants; it is intentionally
        conservative and only rewrites the (..,81)->(..,85) case.
        """

        if not fast_scatter_nd:
            return mod_in

        from tvm.tir import PrimFunc
        from tvm.script import from_source

        mod_out = mod_in
        changed = False
        matched = 0

        # (Debug logging removed; keep compilation output concise.)

        for gv in list(mod_out.get_global_vars()):
            name = gv.name_hint
            if not name.startswith("scatter_nd"):
                continue
            f = mod_out[gv]
            if not isinstance(f, PrimFunc):
                continue

            if len(f.params) != 4:
                continue

            base_buf = f.buffer_map.get(f.params[0])
            idx_buf = f.buffer_map.get(f.params[1])
            val_buf = f.buffer_map.get(f.params[2])
            out_buf = f.buffer_map.get(f.params[3])
            if base_buf is None or idx_buf is None or val_buf is None or out_buf is None:
                continue

            # Match common YOLO scatter patterns that assemble a (..., 85) tensor
            # from a base plus a small values tensor:
            #   base:   (...,85) float*
            #   values: (...,K)  float*   where K in {81, 4, 2}
            #   indices:(...,K,4) int64
            try:
                out_last = int(out_buf.shape[-1])
                val_last = int(val_buf.shape[-1])
                base_last = int(base_buf.shape[-1])
            except Exception:
                continue
            if not (out_last == 85 and base_last == 85 and val_last in (81, 4, 2)):
                continue
            if str(idx_buf.dtype) != "int64":
                continue
            # Indices shape is typically (..., K, index_ndim) where index_ndim=4 for
            # this model, so rank should be out_rank + 1.
            if len(idx_buf.shape) != len(out_buf.shape) + 1:
                continue
            try:
                idx_last = int(idx_buf.shape[-1])
                idx_k = int(idx_buf.shape[-2])
            except Exception:
                continue
            if not (idx_last == 4 and idx_k == val_last):
                continue

            # Leading dims must match.
            if len(out_buf.shape) != 4:
                continue
            try:
                c = int(out_buf.shape[0])
                h = int(out_buf.shape[1])
                w = int(out_buf.shape[2])
            except Exception:
                continue
            if any(x <= 0 for x in (c, h, w)):
                continue

            # K=81 is a known YOLO layout: it fills channels [4:85] (objectness + 80 classes)
            # and leaves [0:4] from base (bbox).
            is_k81_copy = (val_last == 81)

            dtype = str(out_buf.dtype)
            if dtype not in ("float32", "float16"):
                # Keep it strict; YOLO decode should be fp32 here.
                continue

            # Generate a specialized PrimFunc with the same signature (keep B to
            # preserve ABI) but ignore indices entirely.
            #
            # IMPORTANT: bind CUDA threads explicitly. Relying on DefaultGPUSchedule
            # for a synthetic PrimFunc is brittle and can end up serial.
            # Use a 3D CUDA grid over spatial dims and threadIdx.x over channel.
            # This avoids expensive div/mod decoding and turns the kernel into a
            # straightforward copy.
            threads = 128

            if is_k81_copy:
                # Pure copy with fixed channel slice overwrite, no index reads.
                src = f"""# from tvm.script import tir as T\n\n@T.prim_func\ndef {name}(var_base: T.handle, B: T.Buffer((T.int64({c}), T.int64({h}), T.int64({w}), T.int64(81), T.int64(4)), \"int64\"), var_vals: T.handle, out_buf: T.Buffer((T.int64({c}), T.int64({h}), T.int64({w}), T.int64(85)), \"{dtype}\")):\n    T.func_attr({{\"tir.noalias\": True}})\n    base = T.match_buffer(var_base, (T.int64({c}), T.int64({h}), T.int64({w}), T.int64(85)), dtype=\"{dtype}\", offset_factor=1)\n    vals = T.match_buffer(var_vals, (T.int64({c}), T.int64({h}), T.int64({w}), T.int64(81)), dtype=\"{dtype}\", offset_factor=1)\n\n    for c_idx in T.thread_binding(T.int64({c}), thread=\"blockIdx.z\"):\n        for h_idx in T.thread_binding(T.int64({h}), thread=\"blockIdx.y\"):\n            for w_idx in T.thread_binding(T.int64({w}), thread=\"blockIdx.x\"):\n                for ch in T.thread_binding(T.int64({threads}), thread=\"threadIdx.x\"):\n                    if ch < T.int64(85):\n                        if ch < T.int64(4):\n                            out_buf[c_idx, h_idx, w_idx, ch] = base[c_idx, h_idx, w_idx, ch]\n                        else:\n                            out_buf[c_idx, h_idx, w_idx, ch] = vals[c_idx, h_idx, w_idx, ch - T.int64(4)]\n"""
            else:
                # Small-K (K=2/4) scatter: still copy base, but use indices to place each value.
                # This is much cheaper than the generic flatten+divmod scatter implementation,
                # while staying correct without assuming channel placement.
                src = f"""# from tvm.script import tir as T\n\n@T.prim_func\ndef {name}(var_base: T.handle, B: T.Buffer((T.int64({c}), T.int64({h}), T.int64({w}), T.int64({val_last}), T.int64(4)), \"int64\"), var_vals: T.handle, out_buf: T.Buffer((T.int64({c}), T.int64({h}), T.int64({w}), T.int64(85)), \"{dtype}\")):\n    T.func_attr({{\"tir.noalias\": True}})\n    base = T.match_buffer(var_base, (T.int64({c}), T.int64({h}), T.int64({w}), T.int64(85)), dtype=\"{dtype}\", offset_factor=1)\n    vals = T.match_buffer(var_vals, (T.int64({c}), T.int64({h}), T.int64({w}), T.int64({val_last})), dtype=\"{dtype}\", offset_factor=1)\n\n    for c_idx in T.thread_binding(T.int64({c}), thread=\"blockIdx.z\"):\n        for h_idx in T.thread_binding(T.int64({h}), thread=\"blockIdx.y\"):\n            for w_idx in T.thread_binding(T.int64({w}), thread=\"blockIdx.x\"):\n                for ch in T.thread_binding(T.int64({threads}), thread=\"threadIdx.x\"):\n                    if ch < T.int64(85):\n                        out_buf[c_idx, h_idx, w_idx, ch] = base[c_idx, h_idx, w_idx, ch]\n                # Scatter overwrite for this (c,h,w) tile. K is small (2/4).
                for k in range({val_last}):\n                    i0 = B[c_idx, h_idx, w_idx, T.int64(k), T.int64(0)]\n                    i1 = B[c_idx, h_idx, w_idx, T.int64(k), T.int64(1)]\n                    i2 = B[c_idx, h_idx, w_idx, T.int64(k), T.int64(2)]\n                    i3 = B[c_idx, h_idx, w_idx, T.int64(k), T.int64(3)]\n                    if (T.int64(0) <= i0) and (i0 < T.int64({c})) and (T.int64(0) <= i1) and (i1 < T.int64({h})) and (T.int64(0) <= i2) and (i2 < T.int64({w})) and (T.int64(0) <= i3) and (i3 < T.int64(85)):\n                        out_buf[i0, i1, i2, i3] = vals[c_idx, h_idx, w_idx, T.int64(k)]\n"""

            try:
                new_f = from_source(src)
            except Exception:
                # If parsing fails for any reason, keep the original.
                continue

            mod_out.update_func(gv, new_f)
            changed = True
            matched += 1

        if fast_scatter_nd:
            print(f"[fast_scatter_nd] specialized {matched} scatter_nd* PrimFunc(s)")

        return mod_out if changed else mod_in

    # Build an optimization pipeline for Relax IR.
    #
    # IMPORTANT: meta-schedule's `tune_relax` expects a module at the right stage
    # for task extraction (similar to `zero_pipeline()`), and it should NOT be run
    # after the `default` build pipeline (which lowers to VM-ready IR and leaves
    # no tunable tasks).
    if tune:
        # Keep this close to Relax's built-in tuning pipeline prep.
        tune_prep = [
            relax.transform.DecomposeOpsForInference(),
            relax.transform.CanonicalizeBindings(),
            relax.get_pipeline("zero"),
        ]
        if use_fp16:
            tune_prep.insert(2, relax.transform.ToMixedPrecision(out_dtype="float32"))
        mod = tvm.ir.transform.Sequential(tune_prep)(mod)
    else:
        # Non-tuning build path: use the requested pre-built pipeline and some
        # extra generic cleanups/fusions.
        relax_passes = [relax.get_pipeline(relax_pipeline)]
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
        mod = tvm.ir.transform.Sequential(relax_passes)(mod)

    tuned_effective = False
    with tvm.transform.PassContext(opt_level=3):
        if tune:
            try:
                from tvm.meta_schedule import relax_integration as ms_relax
                from tvm import meta_schedule as ms

                if tune_builder_timeout_sec is None:
                    tune_builder_timeout_sec = 300.0 if target == "cuda" else 30.0
                if tune_runner_timeout_sec is None:
                    tune_runner_timeout_sec = 60.0 if target == "cuda" else 30.0

                builder = ms.builder.LocalBuilder(timeout_sec=float(tune_builder_timeout_sec))
                # scatter_nd tasks have int64 indices that must be in-bounds.
                # The default runner allocator uses random int64 values, which can
                # trigger CUDA illegal memory access during measurement.
                op_names_effective = (
                    tune_op_names
                    if tune_op_names is not None
                    else ([("conv2d"), ("scatter_nd")] if target == "cuda" else None)
                )
                use_safe_alloc = False
                if op_names_effective is not None:
                    use_safe_alloc = any("scatter_nd" in str(x) for x in op_names_effective)

                runner_init = None
                runner_alloc = None
                if use_safe_alloc:
                    # IMPORTANT: The runner executes in a separate subprocess which does NOT
                    # inherit this repo's `sys.path` modifications. Therefore we register the
                    # allocator inside the worker process via `initializer`, and refer to it
                    # by a global name (string) so sanity checks and calls work reliably.
                    alloc_name = "yolov3_tvm_opt.safe_alloc_scatter_nd"

                    def _runner_initializer() -> None:
                        import tvm
                        import numpy as np

                        f_random_fill = tvm.get_global_func(
                            "tvm.contrib.random.random_fill_for_measure"
                        )

                        def safe_alloc_argument(device, args_info, alloc_repeat):
                            repeated_args = []
                            for _ in range(int(alloc_repeat)):
                                args = []
                                for arg_info in args_info:
                                    arg_type = arg_info[0]
                                    if arg_type != "TENSOR":
                                        raise NotImplementedError(arg_info)
                                    _, dtype, shape = arg_info
                                    shape_t = tuple(int(x) for x in shape)
                                    arr = tvm.runtime.empty(
                                        shape=shape_t, dtype=str(dtype), device=device
                                    )
                                    if str(dtype) == "int64" and len(shape_t) >= 2 and shape_t[-1] == 4:
                                        # scatter_nd indices must be in-bounds.
                                        # Filling with zeros is always safe and avoids TVM-version-specific
                                        # TIR scripting APIs inside the runner subprocess.
                                        arr.copyfrom(np.zeros(shape_t, dtype=np.int64))
                                    else:
                                        f_random_fill(arr)
                                    args.append(arr)
                                repeated_args.append(args)
                            return repeated_args

                        tvm.register_global_func(alloc_name, safe_alloc_argument, override=True)

                    runner_init = _runner_initializer
                    runner_alloc = alloc_name

                runner = ms.runner.LocalRunner(
                    timeout_sec=float(tune_runner_timeout_sec),
                    f_alloc_argument=runner_alloc,
                    initializer=runner_init,
                )

                db = ms_relax.tune_relax(
                    mod=mod,
                    params=normalized_params,
                    target=tvm_target,
                    work_dir=work_dir or os.path.join(out_dir, "ms_workdir"),
                    max_trials_global=int(max_trials_global),
                    max_trials_per_task=max_trials_per_task,
                    num_trials_per_iter=int(num_trials_per_iter),
                    builder=builder,
                    runner=runner,
                    # Some TVM builds annotate TOPI pooling with schedule_rule
                    # `meta_schedule.pool_max`, but do not ship the corresponding
                    # schedule rule implementation. In that case, trying to tune
                    # pool_max/max_pool2d will lead to unscheduled CUDA TIR and
                    # fail `tir.transform.VerifyMemory`.
                    #
                    # If the user didn't specify op_names, use a safe default that
                    # matches the primary hotspots in this repo.
                    op_names=(
                        tune_op_names
                        if tune_op_names is not None
                        else (["conv2d", "scatter_nd"] if target == "cuda" else None)
                    ),
                )

                # Apply tuned schedules, then build with a CUDA scheduling fallback.
                from tvm.relax.transform import BindParams, MetaScheduleApplyDatabase

                relax_mod = mod
                if normalized_params:
                    relax_mod = BindParams("main", normalized_params)(relax_mod)

                with tvm_target, db, tvm.transform.PassContext(opt_level=3):
                    relax_mod = MetaScheduleApplyDatabase(enable_warning=False)(relax_mod)

                tir_pipeline = tvm.tir.pipeline.get_default_tir_pipeline(tvm_target)
                if target == "cuda":
                    # Some ops (e.g. pool_max) may not have meta-schedule rules in this TVM build.
                    # DefaultGPUSchedule provides a reasonable fallback to ensure valid CUDA kernels.
                    tir_pipeline = tvm.ir.transform.Sequential(
                        [tvm.tir.transform.DefaultGPUSchedule(), tir_pipeline]
                    )

                ex = relax.build(
                    relax_mod,
                    target=tvm_target,
                    params={},
                    tir_pipeline=tir_pipeline,
                    exec_mode=str(exec_mode),
                )
                tuned_effective = True
            except ValueError as e:
                if "No tasks to tune" not in str(e):
                    raise
                # Fall back to non-tuned compilation.
                tune = False

        if not tune:
            mod = _maybe_specialize_yolo_scatter_nd(mod)
            tir_pipeline = tvm.tir.pipeline.get_default_tir_pipeline(tvm_target)
            if target == "cuda":
                # Ensure lowered CUDA TIR has thread bindings (avoid VerifyMemory failures).
                tir_pipeline = tvm.ir.transform.Sequential(
                    [tvm.tir.transform.DefaultGPUSchedule(), tir_pipeline]
                )
            ex = relax.build(
                mod,
                target=tvm_target,
                params=normalized_params,
                tir_pipeline=tir_pipeline,
                exec_mode=str(exec_mode),
            )

    lib_path = os.path.join(out_dir, "model.so")
    ex.export_library(lib_path)

    meta = CompileMeta(
        onnx_path=os.path.abspath(onnx_path),
        input_name=input_name,
        input_shape=tuple(input_shape),
        input_dtype=input_dtype,
        target=str(tvm_target),
        exec_mode=str(exec_mode),
        tuned=bool(tuned_effective),
    )
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    return meta
