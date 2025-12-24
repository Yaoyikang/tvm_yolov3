# yolov3_tvm_opt

一个最小可复用的 Python 项目：把 YOLOv3（ONNX）用 TVM 编译到 CPU/CUDA，并做简单 benchmark。

## 依赖

建议在你的 conda 环境里安装（示例）：

```bash
pip install -r requirements.txt
```

注意：
- TVM 不在 requirements 里安装（你可能用源码编译版）。脚本会优先 `import tvm`，如果失败会尝试用 `TVM_HOME=/root/myfold/tvm` 把 `TVM_HOME/python` 加入 `sys.path`。
- CUDA 跑需要：`tvm.runtime.enabled("cuda") == True`，且 `TVM_LIBRARY_PATH` 指向你编译出来的 runtime（例如 `/root/myfold/tvm/build_cuda`）。

## 使用

### 0) Smoke test（不依赖外部模型）

先确认 TVM 导入/编译/加载链路是通的：

```bash
# 如果你要测 CUDA，请先确保 TVM_LIBRARY_PATH 指向 CUDA build
TVM_LIBRARY_PATH=/root/myfold/tvm/build_cuda python scripts/smoke_test_identity.py
```

### 1) 编译

```bash
# CUDA
TVM_LIBRARY_PATH=/root/myfold/tvm/build_cuda \
python scripts/compile_yolov3.py \
  --onnx /path/to/yolov3.onnx \
  --target cuda \
  --out_dir artifacts/yolov3_cuda

# CPU
python scripts/compile_yolov3.py \
  --onnx /path/to/yolov3.onnx \
  --target llvm \
  --out_dir artifacts/yolov3_cpu
```

可选：如果你的 TVM build 带了 meta-schedule，可以加 `--tune` 做简单调优（耗时较长）：

```bash
TVM_LIBRARY_PATH=/root/myfold/tvm/build_cuda \
python scripts/compile_yolov3.py --onnx /path/to/yolov3.onnx --target cuda --out_dir artifacts/yolov3_cuda_tuned --tune
```

如果你启用了 `--fast_scatter_nd`（针对 YOLO 的 `scatter_nd*` 做专用实现替换），建议把产物固定到：

```bash
TVM_LIBRARY_PATH=/root/myfold/tvm/build_cuda \
python scripts/compile_yolov3.py \
  --onnx /path/to/yolov3.onnx \
  --target cuda \
  --out_dir artifacts/yolov3_cuda_opt_fastscatter \
  --fast_scatter_nd
```

### 2) 运行与测速

```bash
# 随机输入基准
TVM_LIBRARY_PATH=/root/myfold/tvm/build_cuda \
python scripts/bench_yolov3.py --artifact_dir artifacts/yolov3_cuda --device cuda --iters 200

# fast_scatter_nd 版本
TVM_LIBRARY_PATH=/root/myfold/tvm/build_cuda \
python scripts/bench_yolov3.py --artifact_dir artifacts/yolov3_cuda_opt_fastscatter --device cuda --iters 200

# 用图片跑一次（RGB + resize 到模型输入尺寸 + 可选缩放），并把输出保存成 npz
TVM_LIBRARY_PATH=/root/myfold/tvm/build_cuda \
python scripts/bench_yolov3.py \
  --artifact_dir artifacts/yolov3_cuda \
  --device cuda \
  --image /path/to/your.jpg \
  --input_scale 1.0 \
  --iters 1 --warmup 0 \
  --save_outputs artifacts/yolov3_cuda/outputs.npz

# 把输出解码为画框后的图片（需要 --image；会在 resize 后的图片上画框）
TVM_LIBRARY_PATH=/root/myfold/tvm/build_cuda \
python scripts/bench_yolov3.py \
  --artifact_dir artifacts/yolov3_cuda \
  --device cuda \
  --image /path/to/your.jpg \
  --input_scale 1.0 \
  --iters 1 --warmup 0 \
  --save_vis artifacts/yolov3_cuda/vis.jpg \
  --score_thresh 0.25 \
  --nms --nms_iou 0.45

# 也可用 CPU
python scripts/bench_yolov3.py --artifact_dir artifacts/yolov3_cpu --device cpu --iters 200
```

### 3) 用 car.jpg 对比优化前/优化后用时（baseline vs fastscatter）
如何看结果：
 - 关注两次输出中的 `avg latency (ms): ...`（越小越快）
 - 可用“加速比”粗算：`speedup = baseline_latency / fastscatter_latency`
 - 同时对比 `--profile` 的 Top-20：如果优化生效，`scatter_nd*` 往往会从热点列表中消失或显著下降。

复现命令（同一输入图片与参数跑两次）：

```bash
cd /root/myfold/triton_program/lite_llama/yolov3_tvm_opt
export TVM_LIBRARY_PATH=/root/myfold/tvm/build_cuda

# 优化前：baseline
python scripts/bench_yolov3.py \
  --artifact_dir artifacts/yolov3_cuda_baseline \
  --device cuda \
  --image pictures/car.jpg \
  --input_scale 0.0039215686 \
  --iters 200 --warmup 10 \
  --profile --profile_topk 20

# 优化后：fastscatter
python scripts/bench_yolov3.py \
  --artifact_dir artifacts/yolov3_cuda_opt_fastscatter \
  --device cuda \
  --image pictures/car.jpg \
  --input_scale 0.0039215686 \
  --iters 200 --warmup 10 \
  --profile --profile_topk 20
```

示例实测结果（同一机器、同一输入 `pictures/car.jpg`、`iters=200`、`warmup=10`、`input_scale=1/255`）：

- baseline：`avg latency (ms): 45.4979`
  - Top hotspot 主要是 `scatter_nd*`（例如 `scatter_nd5 ~20755us`、`scatter_nd2 ~6256us`、`scatter_nd4/3` 各 ~3-4ms）
- fastscatter：`avg latency (ms): 3.8628`
  - Top hotspot 变为 `conv2d*`，`scatter_nd*` 基本不再进入 Top-20
- 加速比：`45.4979 / 3.8628 ≈ 11.78x`

完整输出（节选）如下，便于复现对比：

```text
# baseline
=== Per-op Profile (Top 20 ) ===
Rank  Op                                          Time(us)
   1  scatter_nd5                                 20755.46
   2  scatter_nd2                                  6256.64
   3  scatter_nd4                                  4008.96
   5  scatter_nd3                                  3464.19
   9  conv2d6                                       947.20
  12  conv2d10                                      712.70
=== End Profile ===

avg latency (ms): 45.4979

# fastscatter
=== Per-op Profile (Top 20 ) ===
Rank  Op                                          Time(us)
   1  conv2d6                                       865.28
   2  conv2d10                                      651.26
   3  conv2d1                                       261.12
   4  conv2d2                                       249.85
   5  conv2d3                                       249.85
=== End Profile ===

avg latency (ms): 3.8628
```

说明：不同导出的 YOLOv3 ONNX 可能期望输入范围不同。
- 如果你的模型期望 [0,255]，用 `--input_scale 1.0`（默认）
- 如果你的模型期望 [0,1]，用 `--input_scale 0.0039215686`（即 1/255）

### 4) 用 cross.jpeg 对比优化前/优化后用时（baseline vs fastscatter）

说明：本对比保持与上面一致的输入预处理（`input_scale=1/255`），并开启 NMS 生成可视化图片，便于直观看到检测效果。

复现命令（同一输入图片与参数跑两次）：

```bash
cd /root/myfold/triton_program/lite_llama/yolov3_tvm_opt
export TVM_LIBRARY_PATH=/root/myfold/tvm/build_cuda

# 优化后：fastscatter（输出 cross_vis2.jpeg）
python scripts/bench_yolov3.py \
  --artifact_dir artifacts/yolov3_cuda_opt_fastscatter \
  --device cuda \
  --image pictures/cross.jpeg \
  --input_scale 0.0039215686 \
  --save_vis pictures/cross_vis2.jpeg \
  --nms --nms_iou 0.45 \
  --iters 1 --warmup 0

# 优化前：baseline（输出 cross_vis.jpeg）
python scripts/bench_yolov3.py \
  --artifact_dir artifacts/yolov3_cuda_baseline \
  --device cuda \
  --image pictures/cross.jpeg \
  --input_scale 0.0039215686 \
  --save_vis pictures/cross_vis.jpeg \
  --nms --nms_iou 0.45 \
  --iters 1 --warmup 0
```

示例实测结果（同一机器、同一输入、参数如上）：

- baseline：`avg latency (ms): 51.7521`，输出：`pictures/cross_vis.jpeg`
- fastscatter：`avg latency (ms): 5.6705`，输出：`pictures/cross_vis2.jpeg`
- 加速比：`51.7521 / 5.6705 ≈ 9.13x`

## 说明

`--out_dir` 下会生成一个可直接运行的 Relax VM artifact 目录，典型结构如下：

```text
artifacts/xxx/
  model.so
  meta.json
```

文件含义：

- `model.so`
  - TVM 编译生成的动态库，里面包含 Relax VMExecutable 以及该模型用到的算子实现（在 CUDA target 下也包含生成的 GPU kernel）。
  - 运行时通过 `tvm.runtime.load_module(model.so)` 加载，并用 `relax.vm_build.VMExecutable(lib)` + `relax.VirtualMachine(...)` 执行。
  - 该文件通常和编译环境强相关：TVM 版本、CUDA/驱动版本、GPU 架构（例如 `sm_89`）等变化都可能需要重新编译。

- `meta.json`
  - 该工程写入的元信息（用于运行脚本自动构造输入、选择正确 target 等）。典型字段：
    - `onnx_path`：编译时使用的 ONNX 路径（绝对路径，仅用于记录）
    - `input_name`：模型输入 tensor 名称（例如 `image`）
    - `input_shape`：输入形状 NCHW（例如 `[1,3,640,640]`）
    - `input_dtype`：输入 dtype（例如 `float32`）
    - `target`：实际编译 target 字符串（CUDA 下通常包含 arch 和线程/共享内存等约束）
    - `exec_mode`：Relax VM 执行模式（`bytecode` 或 `compiled`）
    - `tuned`：是否应用了 meta-schedule 的调优数据库（如果你启用 `--tune` 并成功应用则为 true）

如何运行这些：

- 最简单：直接用脚本加载并运行
  - `python scripts/bench_yolov3.py --artifact_dir <out_dir> --device cuda ...`

- 代码里加载（见 `src/yolov3_tvm_opt/runtime.py` 的 `load_artifact/create_vm`）
  - `load_artifact(out_dir)` 会读取 `meta.json` 并定位 `model.so`
  - `create_vm(artifact, device=..., profile=...)` 会创建 `relax.VirtualMachine`

可复现/可移植性注意：

- `model.so` 一般不能“保证跨环境通用”。如果你换了 GPU（不同 SM）、换了 TVM build、换了 CUDA/驱动版本，建议重新跑 `scripts/compile_yolov3.py` 生成新的 artifact。
- 本仓库默认把 `artifacts/` 加入 `.gitignore`（避免提交大文件/二进制）。如果你确实希望提交某个 artifact，需要手动调整 `.gitignore` 或显式添加。
