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

### 2) 运行与测速

```bash
# 随机输入基准
TVM_LIBRARY_PATH=/root/myfold/tvm/build_cuda \
python scripts/bench_yolov3.py --artifact_dir artifacts/yolov3_cuda --device cuda --iters 200

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

说明：不同导出的 YOLOv3 ONNX 可能期望输入范围不同。
- 如果你的模型期望 [0,255]，用 `--input_scale 1.0`（默认）
- 如果你的模型期望 [0,1]，用 `--input_scale 0.0039215686`（即 1/255）

# 也可用 CPU
python scripts/bench_yolov3.py --artifact_dir artifacts/yolov3_cpu --device cpu --iters 200
```

## 产物说明

`--out_dir` 下会生成：
- `model.so`：Relax VMExecutable（可被 `tvm.runtime.load_module` 加载）
- `meta.json`：记录输入 shape/target 等元信息
