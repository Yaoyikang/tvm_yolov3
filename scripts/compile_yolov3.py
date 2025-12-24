#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from yolov3_tvm_opt.compile_relay import compile_relay


def main() -> None:
    p = argparse.ArgumentParser("Compile YOLOv3 ONNX with TVM (Relay)")
    p.add_argument("--onnx", required=True, help="Path to yolov3.onnx")
    p.add_argument("--out_dir", required=True, help="Output directory for artifacts")
    p.add_argument("--target", default="cuda", choices=["cuda", "llvm"], help="Compilation target")
    p.add_argument("--input_name", default=None, help="Override ONNX input name")
    p.add_argument(
        "--input_shape",
        default=None,
        help="Override input shape NCHW, e.g. 1,3,416,416",
    )
    p.add_argument("--tune", action="store_true", help="Enable meta-schedule tuning (slow)")
    p.add_argument("--work_dir", default=None, help="Meta-schedule work dir (if --tune)")
    p.add_argument("--max_trials_global", type=int, default=2000)
    p.add_argument("--max_trials_per_task", type=int, default=0, help="0 means unset")
    p.add_argument("--num_trials_per_iter", type=int, default=64)
    p.add_argument(
        "--tune_builder_timeout_sec",
        type=float,
        default=None,
        help="Meta-schedule LocalBuilder timeout (seconds). If omitted, defaults to 300s on CUDA, 30s otherwise.",
    )
    p.add_argument(
        "--tune_runner_timeout_sec",
        type=float,
        default=None,
        help="Meta-schedule LocalRunner timeout (seconds). If omitted, defaults to 60s on CUDA, 30s otherwise.",
    )
    p.add_argument(
        "--tune_op_names",
        default=None,
        help=(
            "Comma-separated substrings to select which tasks to tune, e.g. 'conv2d,scatter_nd'. "
            "If omitted on CUDA, defaults to 'conv2d,scatter_nd' to avoid missing pool_max rule in some TVM builds."
        ),
    )
    p.add_argument(
        "--relax_pipeline",
        default="default",
        choices=["zero", "default", "default_build", "static_shape_tuning"],
        help="Relax pre-built pipeline name",
    )
    p.add_argument(
        "--exec_mode",
        default="bytecode",
        choices=["bytecode", "compiled"],
        help="Relax VM execution mode. 'compiled' can reduce VM overhead.",
    )
    p.add_argument("--fp16", action="store_true", help="Enable mixed precision (fp16 compute where possible)")
    p.add_argument(
        "--fast_scatter_nd",
        action="store_true",
        help="Specialize YOLO scatter_nd (..,81)->(..,85) into pure copies to avoid expensive scatter kernels.",
    )
    args = p.parse_args()

    input_shape = None
    if args.input_shape:
        parts = [int(x) for x in args.input_shape.split(",")]
        if len(parts) != 4:
            raise ValueError("--input_shape must be N,C,H,W")
        input_shape = (parts[0], parts[1], parts[2], parts[3])

    max_trials_per_task = None if args.max_trials_per_task == 0 else int(args.max_trials_per_task)

    tune_op_names = None
    if args.tune_op_names:
        tune_op_names = [s.strip() for s in str(args.tune_op_names).split(",") if s.strip()]

    meta = compile_relay(
        onnx_path=args.onnx,
        out_dir=args.out_dir,
        target=args.target,
        input_name=args.input_name,
        input_shape=input_shape,
        tune=bool(args.tune),
        work_dir=args.work_dir,
        max_trials_global=int(args.max_trials_global),
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=int(args.num_trials_per_iter),
        use_fp16=bool(args.fp16),
        fast_scatter_nd=bool(args.fast_scatter_nd),
        relax_pipeline=str(args.relax_pipeline),
        exec_mode=str(args.exec_mode),
        tune_op_names=tune_op_names,
        tune_builder_timeout_sec=args.tune_builder_timeout_sec,
        tune_runner_timeout_sec=args.tune_runner_timeout_sec,
    )

    print("OK: compiled")
    print("out_dir:", os.path.abspath(args.out_dir))
    print("input:", meta.input_name, meta.input_shape, meta.input_dtype)
    print("target:", meta.target)
    print("tuned:", meta.tuned)


if __name__ == "__main__":
    main()
