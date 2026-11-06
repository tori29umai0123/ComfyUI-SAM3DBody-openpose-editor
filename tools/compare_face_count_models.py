from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ANIMEHEAD_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
RTDETR_FACE_THRESHOLDS = (0.03, 0.05, 0.07, 0.10, 0.15, 0.20)


def imread(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path: Path, image: np.ndarray) -> None:
    ok, enc = cv2.imencode(".png", image)
    if ok:
        enc.tofile(str(path))


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
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return keep


def letterbox_bgr(image: np.ndarray, size: int = 640):
    h, w = image.shape[:2]
    scale = min(size / max(h, 1), size / max(w, 1))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px = (size - nw) // 2
    py = (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, scale, px, py


def run_animeheads(net, image: np.ndarray, threshold: float) -> list[dict]:
    input_image, scale, pad_x, pad_y = letterbox_bgr(image, 640)
    blob = cv2.dnn.blobFromImage(input_image, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()
    if output.ndim == 3:
        output = output[0]
    if output.shape[0] == 5:
        output = output.T
    ih, iw = image.shape[:2]
    boxes_xywh = []
    scores = []
    for row in output:
        if len(row) < 5:
            continue
        cx, cy, bw, bh, score = [float(v) for v in row[:5]]
        if score < threshold:
            continue
        x1 = (cx - bw * 0.5 - pad_x) / scale
        y1 = (cy - bh * 0.5 - pad_y) / scale
        x2 = (cx + bw * 0.5 - pad_x) / scale
        y2 = (cy + bh * 0.5 - pad_y) / scale
        x1, x2 = np.clip([x1, x2], 0, iw - 1)
        y1, y2 = np.clip([y1, y2], 0, ih - 1)
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            continue
        boxes_xywh.append([int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))])
        scores.append(score)
    if not boxes_xywh:
        return []
    keep = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=threshold, nms_threshold=0.45)
    detections = []
    for i in np.array(keep).reshape(-1).tolist():
        x, y, bw, bh = boxes_xywh[i]
        detections.append({"label": "animehead", "score": float(scores[i]), "box": [float(x), float(y), float(x + bw), float(y + bh)]})
    return detections


def resize_rtdetrv4(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (1280, 1280), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def run_rtdetrv4_face(session: ort.InferenceSession, image: np.ndarray, threshold: float) -> list[dict]:
    h, w = image.shape[:2]
    blob = resize_rtdetrv4(image)
    labels, boxes, scores = session.run(None, {"images": blob, "orig_target_sizes": np.asarray([[w, h]], dtype=np.int64)})
    labels = labels[0]
    boxes = boxes[0].astype(np.float32)
    scores = scores[0].astype(np.float32)
    keep_mask = (labels == 3) & (scores >= threshold)
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    if len(boxes) == 0:
        return []
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    valid = (boxes[:, 2] > boxes[:, 0] + 2) & (boxes[:, 3] > boxes[:, 1] + 2)
    boxes = boxes[valid]
    scores = scores[valid]
    keep = nms(boxes, scores, 0.45)
    return [{"label": "rtdetrv4_face", "score": float(scores[i]), "box": boxes[i].tolist()} for i in keep]


def draw(image: np.ndarray, detections: list[dict], title: str) -> np.ndarray:
    out = image.copy()
    for idx, det in enumerate(detections, 1):
        x0, y0, x1, y1 = [int(round(v)) for v in det["box"]]
        color = (0, 0, 255) if det["label"] == "animehead" else (255, 0, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 4)
        cv2.putText(out, f"{idx}:{det['score']:.2f}", (x0, max(24, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(out, f"{title} count={len(detections)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def write_csv(path: Path, detections: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "score", "x0", "y0", "x1", "y1"])
        for idx, det in enumerate(detections, 1):
            writer.writerow([idx, det["label"], f"{det['score']:.6f}", *[f"{v:.2f}" for v in det["box"]]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--animeheads-model", default=r"C:\ComfyUI\models\animeheads\animeheadsv3.onnx")
    parser.add_argument("--cache-dir", default=r"C:\tmp\hf_bbox_cache")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mkdir(output_dir)
    images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    animeheads_net = cv2.dnn.readNetFromONNX(str(Path(args.animeheads_model)))
    rtdetr_path = hf_hub_download("tori29umai/rtdetrv4-x-manga109s_v2", "model.onnx", cache_dir=args.cache_dir)
    rtdetr_session = ort.InferenceSession(rtdetr_path, providers=["CPUExecutionProvider"])
    summary = []

    for model_name, thresholds, runner in [
        ("animeheads_v3", ANIMEHEAD_THRESHOLDS, lambda img, thr: run_animeheads(animeheads_net, img, thr)),
        ("rtdetrv4_x_manga109s_v2_face", RTDETR_FACE_THRESHOLDS, lambda img, thr: run_rtdetrv4_face(rtdetr_session, img, thr)),
    ]:
        for thr in thresholds:
            out_dir = output_dir / model_name / f"thr_{thr:.2f}"
            mkdir(out_dir)
            for image_path in images:
                image = imread(image_path)
                if image is None:
                    continue
                detections = runner(image, thr)
                imwrite(out_dir / f"{image_path.stem}_face_bbox.png", draw(image, detections, f"{model_name} thr={thr:.2f}"))
                write_csv(out_dir / f"{image_path.stem}_detections.csv", detections)
                summary.append([model_name, f"{thr:.2f}", image_path.name, len(detections)])

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "threshold", "image", "count"])
        writer.writerows(summary)


if __name__ == "__main__":
    main()
