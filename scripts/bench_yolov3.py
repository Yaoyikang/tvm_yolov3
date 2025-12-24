#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw


class _PreprocessMeta:
    def __init__(self, orig_w: int, orig_h: int, model_w: int, model_h: int, scale: float, pad_left: int, pad_top: int):
        self.orig_w = int(orig_w)
        self.orig_h = int(orig_h)
        self.model_w = int(model_w)
        self.model_h = int(model_h)
        self.scale = float(scale)
        self.pad_left = int(pad_left)
        self.pad_top = int(pad_top)

_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from yolov3_tvm_opt.runtime import benchmark, create_vm, load_artifact


def _preprocess_image(
    image_path: str,
    n: int,
    c: int,
    h: int,
    w: int,
    dtype: str,
    input_scale: float,
    swap_rb: bool,
) -> tuple[np.ndarray, _PreprocessMeta]:
    if n != 1:
        raise ValueError(f"Only batch=1 image supported for --image, got N={n}")
    if c != 3:
        raise ValueError(f"Only 3-channel input supported for --image, got C={c}")

    img0 = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img0.size

    # Letterbox: keep aspect ratio, pad to model size.
    scale = min(float(w) / float(orig_w), float(h) / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    img = img0.resize((new_w, new_h), resample=Image.BILINEAR)

    pad_left = (w - new_w) // 2
    pad_top = (h - new_h) // 2
    canvas = Image.new("RGB", (w, h), color=(128, 128, 128))
    canvas.paste(img, (pad_left, pad_top))

    arr = np.asarray(canvas, dtype=np.float32)  # HWC, RGB (letterboxed)
    # Many ONNX exports expect either [0,1] (scale=1/255) or [0,255] (scale=1.0).
    arr = arr * float(input_scale)
    if swap_rb:
        # RGB <-> BGR
        arr = arr[:, :, ::-1]
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, axis=0)  # NCHW

    # Respect compiled input dtype.
    if dtype == "float16":
        arr = arr.astype(np.float16)
    else:
        arr = arr.astype(np.float32)

    meta = _PreprocessMeta(
        orig_w=orig_w,
        orig_h=orig_h,
        model_w=w,
        model_h=h,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    return arr, meta


def _ensure_dir_for_file(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)


def _boxes_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Best-effort: accept xyxy or cxcywh."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes shape [N,4], got {boxes.shape}")
    b = boxes.astype(np.float32, copy=False)

    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    invalid_frac = float(np.mean((x2 < x1) | (y2 < y1))) if b.shape[0] else 0.0
    if invalid_frac > 0.5:
        # Likely cx,cy,w,h
        cx, cy, w, h = x1, y1, x2, y2
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        b = np.stack([x1, y1, x2, y2], axis=1)
    return b


def _maybe_scale_boxes_xyxy(boxes_xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    b = boxes_xyxy.astype(np.float32, copy=False)
    vmax = float(np.max(b))
    # Heuristic: treat as normalized if values are around [0,1].
    if vmax <= 1.5:
        b = b.copy()
        b[:, [0, 2]] *= float(w)
        b[:, [1, 3]] *= float(h)
    return b


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy NMS for xyxy boxes. Returns indices into the input arrays."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)
        order = rest[iou < float(iou_thresh)]

    return np.asarray(keep, dtype=np.int64)


def _draw_boxes(
    image_path: str,
    save_path: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    model_w: int,
    model_h: int,
    score_thresh: float,
    max_dets: int,
    nms: bool,
    nms_iou: float,
    class_ids: list[int] | None,
    preprocess: _PreprocessMeta,
) -> None:
    # Visualize on the ORIGINAL image, mapping boxes back from letterboxed model space.
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    boxes = np.asarray(boxes)
    scores = np.asarray(scores).reshape(-1)
    classes = np.asarray(classes).reshape(-1)

    if boxes.ndim == 3:
        boxes = boxes[0]
    if scores.ndim == 2:
        scores = scores[0]
    if classes.ndim == 2:
        classes = classes[0]

    boxes = _boxes_to_xyxy(boxes)
    boxes = _maybe_scale_boxes_xyxy(boxes, w=model_w, h=model_h)

    # Clamp to image bounds before any filtering/NMS.
    boxes = boxes.copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(model_w - 1))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(model_h - 1))

    # Map from model (letterboxed) coords -> original image coords.
    if preprocess.scale <= 0.0:
        raise ValueError(f"Invalid preprocess scale: {preprocess.scale}")
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - float(preprocess.pad_left)) / float(preprocess.scale)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - float(preprocess.pad_top)) / float(preprocess.scale)

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(preprocess.orig_w - 1))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(preprocess.orig_h - 1))

    keep = scores >= float(score_thresh)
    idx = np.where(keep)[0]

    if class_ids is not None:
        class_ids_set = set(int(x) for x in class_ids)
        idx = idx[np.isin(classes[idx].astype(np.int64, copy=False), list(class_ids_set))]

    if idx.size == 0:
        _ensure_dir_for_file(save_path)
        img.save(save_path)
        return

    # Optional NMS (class-aware)
    if nms:
        kept_all = []
        cls_ids = classes[idx].astype(np.int64, copy=False)
        for cls in np.unique(cls_ids):
            sel = idx[cls_ids == cls]
            k = _nms_xyxy(boxes[sel], scores[sel], iou_thresh=float(nms_iou))
            kept_all.append(sel[k])
        idx = np.concatenate(kept_all, axis=0) if kept_all else np.zeros((0,), dtype=np.int64)
        if idx.size == 0:
            _ensure_dir_for_file(save_path)
            img.save(save_path)
            return

    # sort by score desc (after NMS)
    order = idx[np.argsort(scores[idx])[::-1]]
    order = order[: int(max_dets)]

    for i in order:
        x1, y1, x2, y2 = boxes[i].tolist()
        x1 = max(0.0, min(float(preprocess.orig_w - 1), float(x1)))
        y1 = max(0.0, min(float(preprocess.orig_h - 1), float(y1)))
        x2 = max(0.0, min(float(preprocess.orig_w - 1), float(x2)))
        y2 = max(0.0, min(float(preprocess.orig_h - 1), float(y2)))
        if x2 <= x1 or y2 <= y1:
            continue

        # Minimal styling: red box + simple text
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        txt = f"{int(classes[i])}:{float(scores[i]):.2f}"
        # draw a small filled background for readability
        tx, ty = x1, max(0.0, y1 - 10)
        draw.rectangle([tx, ty, tx + 60, ty + 10], fill=(255, 0, 0))
        draw.text((tx + 2, ty), txt, fill=(255, 255, 255))

    _ensure_dir_for_file(save_path)
    img.save(save_path)


def main() -> None:
    p = argparse.ArgumentParser("Benchmark compiled YOLOv3 TVM artifact")
    p.add_argument("--artifact_dir", required=True, help="Directory produced by compile_yolov3.py")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Runtime device")
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--image",
        default=None,
        help="Optional: path to an image file. If set, uses image preprocessing (RGB + resize + scaling) instead of random input.",
    )
    p.add_argument(
        "--input_scale",
        type=float,
        default=1.0,
        help="Scale factor applied to uint8 RGB pixels after resize. Use 1.0 for [0,255] models; use 1/255 for [0,1] models.",
    )
    p.add_argument("--swap_rb", action="store_true", help="Swap RGB to BGR (or vice versa) in preprocessing")
    p.add_argument(
        "--save_outputs",
        default=None,
        help="Optional: save outputs to a .npz file (or a directory where outputs.npz will be written).",
    )
    p.add_argument(
        "--save_vis",
        default=None,
        help="Optional: save a visualization image with bounding boxes (requires --image).",
    )
    p.add_argument("--score_thresh", type=float, default=0.25, help="Score threshold for visualization")
    p.add_argument("--max_dets", type=int, default=100, help="Max detections to draw")
    p.add_argument("--nms", action="store_true", help="Enable NMS (class-aware) before drawing")
    p.add_argument("--nms_iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument(
        "--class_id",
        default=None,
        help="Optional: only draw these class ids (comma-separated), e.g. '16' for dog in COCO-80.",
    )
    args = p.parse_args()

    artifact = load_artifact(args.artifact_dir)
    meta = artifact.meta

    input_name = meta["input_name"]
    n, c, h, w = [int(x) for x in meta["input_shape"]]

    input_dtype = meta.get("input_dtype", "float32")
    preprocess_meta = None
    if args.image:
        x, preprocess_meta = _preprocess_image(
            args.image,
            n=n,
            c=c,
            h=h,
            w=w,
            dtype=input_dtype,
            input_scale=float(args.input_scale),
            swap_rb=bool(args.swap_rb),
        )
    else:
        x = np.random.randn(n, c, h, w).astype(input_dtype)

    vm, dev, rt = create_vm(artifact, device=args.device, device_id=args.device_id)
    avg_ms, outputs = benchmark(vm, rt, dev, input_name=input_name, input_data=x, warmup=args.warmup, iters=args.iters)

    print("avg latency (ms):", round(avg_ms, 4))
    print("num outputs:", len(outputs))
    for k, v in outputs.items():
        print("output", k, "shape", tuple(v.shape), "dtype", v.dtype)

    if args.save_outputs:
        save_path = args.save_outputs
        if save_path.endswith(os.sep) or os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, "outputs.npz")
        else:
            _ensure_dir_for_file(save_path)

        np.savez(
            save_path,
            **{f"out_{k}": v for k, v in outputs.items()},
            input=x,
            image_path=(args.image or ""),
            input_name=input_name,
            input_shape=np.array([n, c, h, w], dtype=np.int64),
            input_dtype=np.array([input_dtype]),
            swap_rb=np.array([1 if args.swap_rb else 0], dtype=np.int64),
            preprocess=np.array(
                [
                    preprocess_meta.orig_w if preprocess_meta else -1,
                    preprocess_meta.orig_h if preprocess_meta else -1,
                    preprocess_meta.scale if preprocess_meta else -1.0,
                    preprocess_meta.pad_left if preprocess_meta else -1,
                    preprocess_meta.pad_top if preprocess_meta else -1,
                ],
                dtype=np.float32,
            ),
        )
        print("saved outputs to:", os.path.abspath(save_path))

    if args.save_vis:
        if not args.image:
            raise ValueError("--save_vis requires --image")
        # Best-effort expectation for this YOLOv3 export: 3 outputs -> boxes, scores, classes.
        if not ("0" in outputs and "1" in outputs and "2" in outputs):
            raise ValueError(f"--save_vis expects 3 outputs (0/1/2). Got keys: {list(outputs.keys())}")
        class_ids = None
        if args.class_id:
            class_ids = [int(x) for x in str(args.class_id).split(",") if x.strip()]
        if preprocess_meta is None:
            raise ValueError("Internal error: preprocess_meta is None while --image is set")
        _draw_boxes(
            image_path=args.image,
            save_path=args.save_vis,
            boxes=outputs["0"],
            scores=outputs["1"],
            classes=outputs["2"],
            model_w=w,
            model_h=h,
            score_thresh=float(args.score_thresh),
            max_dets=int(args.max_dets),
            nms=bool(args.nms),
            nms_iou=float(args.nms_iou),
            class_ids=class_ids,
            preprocess=preprocess_meta,
        )
        print("saved visualization to:", os.path.abspath(args.save_vis))


if __name__ == "__main__":
    main()
