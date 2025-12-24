#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from yolov3_tvm_opt.compile_relay import compile_relay
from yolov3_tvm_opt.runtime import benchmark, create_vm, load_artifact


def build_identity_onnx(path: str, shape=(1, 3, 416, 416), dtype=np.float32) -> None:
    import onnx
    from onnx import TensorProto, helper

    input_name = "input"
    output_name = "output"

    onnx_dtype = TensorProto.FLOAT if dtype == np.float32 else TensorProto.FLOAT

    inp = helper.make_tensor_value_info(input_name, onnx_dtype, list(shape))
    out = helper.make_tensor_value_info(output_name, onnx_dtype, list(shape))

    node = helper.make_node("Identity", inputs=[input_name], outputs=[output_name])
    graph = helper.make_graph([node], "identity", [inp], [out])
    model = helper.make_model(graph, producer_name="yolov3_tvm_opt_smoke")
    onnx.save(model, path)


def main() -> None:
    out_dir = str(_ROOT / "artifacts" / "smoke_identity")
    onnx_path = str(_ROOT / "artifacts" / "identity.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    build_identity_onnx(onnx_path)

    target = "cuda" if os.environ.get("TVM_LIBRARY_PATH") else "llvm"
    compile_relay(onnx_path=onnx_path, out_dir=out_dir, target=target)

    artifact = load_artifact(out_dir)
    device = "cuda" if "cuda" in str(artifact.meta.get("target", "")) else "cpu"
    vm, dev, rt = create_vm(artifact, device=device)

    n, c, h, w = artifact.meta["input_shape"]
    x = np.random.randn(int(n), int(c), int(h), int(w)).astype("float32")

    avg_ms, outputs = benchmark(vm, rt, dev, artifact.meta["input_name"], x, warmup=3, iters=20)
    print("OK smoke test")
    print("target:", artifact.meta["target"])
    print("device:", device)
    print("avg_ms:", round(avg_ms, 4))
    print("out0 shape:", outputs["0"].shape)


if __name__ == "__main__":
    main()
