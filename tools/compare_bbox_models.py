from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        denom = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / denom
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return keep


def draw_detections(image: np.ndarray, detections: list[dict], title: str) -> np.ndarray:
    out = image.copy()
    palette = [
        (0, 128, 255),
        (0, 255, 128),
        (255, 128, 0),
        (255, 0, 255),
        (80, 220, 255),
        (180, 80, 255),
        (255, 255, 0),
        (0, 255, 255),
    ]
    for idx, det in enumerate(detections, 1):
        x0, y0, x1, y1 = [int(round(v)) for v in det["box"]]
        color = palette[(idx - 1) % len(palette)]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 4)
        label = f"{idx}:{det['score']:.2f}"
        cv2.putText(out, label, (x0, max(24, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(out, f"{title} count={len(detections)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def write_csv(path: Path, detections: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "score", "x0", "y0", "x1", "y1"])
        for idx, det in enumerate(detections, 1):
            writer.writerow([idx, det["label"], f"{det['score']:.6f}", *[f"{v:.2f}" for v in det["box"]]])


def letterbox_yolox(img: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=cv2.INTER_CUBIC)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    blob = padded.transpose(2, 0, 1).astype(np.float32)
    return np.ascontiguousarray(blob), ratio


def yolox_postprocess(outputs: np.ndarray, img_size: tuple[int, int]) -> np.ndarray:
    grids = []
    expanded_strides = []
    for stride in (8, 16, 32):
        hsize, wsize = img_size[0] // stride, img_size[1] // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def run_ksasao(session: ort.InferenceSession, image: np.ndarray, threshold: float) -> list[dict]:
    input_size = (640, 640)
    blob, ratio = letterbox_yolox(image, input_size)
    raw = session.run(None, {session.get_inputs()[0].name: blob[None]})[0]
    pred = yolox_postprocess(raw, input_size)[0]
    boxes = pred[:, :4]
    scores = (pred[:, 4:5] * pred[:, 5:]).reshape(-1)
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    xyxy /= ratio
    keep_mask = scores >= threshold
    xyxy = xyxy[keep_mask]
    scores = scores[keep_mask]
    keep = nms(xyxy, scores, 0.45)
    h, w = image.shape[:2]
    detections = []
    for i in keep:
        box = xyxy[i].copy()
        box[[0, 2]] = np.clip(box[[0, 2]], 0, w - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, h - 1)
        detections.append({"label": "character", "score": float(scores[i]), "box": box.tolist()})
    return detections


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def resize_dfine(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def run_dfine(session: ort.InferenceSession, image: np.ndarray, threshold: float) -> list[dict]:
    h, w = image.shape[:2]
    blob = resize_dfine(image)
    logits, pred_boxes = session.run(None, {session.get_inputs()[0].name: blob})
    person_scores = sigmoid(logits[0, :, 0])
    boxes = pred_boxes[0]
    keep_mask = person_scores >= threshold
    boxes = boxes[keep_mask]
    scores = person_scores[keep_mask]
    if len(boxes) == 0:
        return []
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * w
    xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * h
    xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * w
    xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * h
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h - 1)
    valid = (xyxy[:, 2] > xyxy[:, 0] + 2) & (xyxy[:, 3] > xyxy[:, 1] + 2)
    xyxy = xyxy[valid]
    scores = scores[valid]
    keep = nms(xyxy, scores, 0.55)
    return [{"label": "person", "score": float(scores[i]), "box": xyxy[i].tolist()} for i in keep]



def resize_rtdetrv4(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (1280, 1280), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def run_rtdetrv4_body(session: ort.InferenceSession, image: np.ndarray, threshold: float) -> list[dict]:
    h, w = image.shape[:2]
    blob = resize_rtdetrv4(image)
    orig_target_sizes = np.asarray([[w, h]], dtype=np.int64)
    labels, boxes, scores = session.run(None, {"images": blob, "orig_target_sizes": orig_target_sizes})
    labels = labels[0]
    boxes = boxes[0].astype(np.float32)
    scores = scores[0].astype(np.float32)
    keep_mask = (labels == 0) & (scores >= threshold)
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    if len(boxes) == 0:
        return []
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    valid = (boxes[:, 2] > boxes[:, 0] + 2) & (boxes[:, 3] > boxes[:, 1] + 2)
    boxes = boxes[valid]
    scores = scores[valid]
    keep = nms(boxes, scores, 0.55)
    return [{"label": "body", "score": float(scores[i]), "box": boxes[i].tolist()} for i in keep]


def prepare_models(cache_dir: Path, ksasao_repo: Path) -> dict[str, tuple[ort.InferenceSession, str]]:
    providers = ["CPUExecutionProvider"]
    ksasao_model = ksasao_repo / "python" / "character.onnx"
    dfine_s = hf_hub_download("onnx-community/dfine_s_coco-ONNX", "onnx/model.onnx", cache_dir=str(cache_dir))
    dfine_m = hf_hub_download("onnx-community/dfine_m_coco-ONNX", "onnx/model.onnx", cache_dir=str(cache_dir))
    rtdetrv4 = hf_hub_download("tori29umai/rtdetrv4-x-manga109s_v2", "model.onnx", cache_dir=str(cache_dir))
    return {
        "ksasao_anime_character_detector": (ort.InferenceSession(str(ksasao_model), providers=providers), "ksasao"),
        "dfine_s_coco_person": (ort.InferenceSession(dfine_s, providers=providers), "dfine"),
        "dfine_m_coco_person": (ort.InferenceSession(dfine_m, providers=providers), "dfine"),
        "rtdetrv4_x_manga109s_v2_body": (ort.InferenceSession(rtdetrv4, providers=providers), "rtdetrv4_body"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ksasao-repo", default=r"C:\tmp\anime-character-detector")
    parser.add_argument("--cache-dir", default=r"C:\tmp\hf_bbox_cache")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    mkdir(output_dir)
    models = prepare_models(cache_dir, Path(args.ksasao_repo))
    images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    summary_rows = []

    for model_name, (session, kind) in models.items():
        model_dir = output_dir / model_name
        mkdir(model_dir)
        for thr in THRESHOLDS:
            thr_dir = model_dir / f"thr_{thr:.2f}"
            mkdir(thr_dir)
            for image_path in images:
                image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                if kind == "ksasao":
                    detections = run_ksasao(session, image, thr)
                elif kind == "dfine":
                    detections = run_dfine(session, image, thr)
                elif kind == "rtdetrv4_body":
                    detections = run_rtdetrv4_body(session, image, thr)
                else:
                    raise ValueError(f"Unsupported model kind: {kind}")
                stem = image_path.stem
                annotated = draw_detections(image, detections, f"{model_name} thr={thr:.2f}")
                ok, encoded = cv2.imencode(".png", annotated)
                if ok:
                    encoded.tofile(str(thr_dir / f"{stem}_bbox.png"))
                write_csv(thr_dir / f"{stem}_detections.csv", detections)
                summary_rows.append([model_name, f"{thr:.2f}", image_path.name, len(detections)])

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "threshold", "image", "count"])
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
