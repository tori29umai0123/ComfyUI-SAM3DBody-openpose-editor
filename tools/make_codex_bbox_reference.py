from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MODEL_CHOICES = [
    ("rtdetrv4_x_manga109s_v2_body", "thr_0.10", 1.40),
    ("rtdetrv4_x_manga109s_v2_body", "thr_0.05", 0.70),
    ("ksasao_anime_character_detector", "thr_0.10", 1.10),
    ("ksasao_anime_character_detector", "thr_0.05", 0.60),
    ("dfine_m_coco_person", "thr_0.15", 0.80),
    ("dfine_m_coco_person", "thr_0.10", 0.35),
]


def imread(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path: Path, image: np.ndarray) -> None:
    ok, enc = cv2.imencode(".png", image)
    if ok:
        enc.tofile(str(path))


def read_detections(path: Path, model: str, weight: float) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            box = [float(row[k]) for k in ("x0", "y0", "x1", "y1")]
            score = float(row["score"])
            rows.append({"box": box, "score": score, "model": model, "weight": weight})
    return rows


def iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ba = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / (aa + ba - inter + 1e-6)


def center_inside(a: list[float], b: list[float]) -> bool:
    cx = (a[0] + a[2]) * 0.5
    cy = (a[1] + a[3]) * 0.5
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def cluster_candidates(candidates: list[dict]) -> list[dict]:
    clusters: list[dict] = []
    for cand in sorted(candidates, key=lambda d: d["score"] * d["weight"], reverse=True):
        placed = False
        for cluster in clusters:
            overlap = iou(cand["box"], cluster["box"])
            cand_area = max(1.0, (cand["box"][2] - cand["box"][0]) * (cand["box"][3] - cand["box"][1]))
            cluster_area = max(1.0, (cluster["box"][2] - cluster["box"][0]) * (cluster["box"][3] - cluster["box"][1]))
            size_ratio = min(cand_area, cluster_area) / max(cand_area, cluster_area)
            same_object = overlap >= 0.25 or (overlap >= 0.08 and size_ratio >= 0.35 and (center_inside(cand["box"], cluster["box"]) or center_inside(cluster["box"], cand["box"])))
            if same_object:
                cluster["items"].append(cand)
                boxes = np.asarray([item["box"] for item in cluster["items"]], dtype=np.float32)
                weights = np.asarray([item["score"] * item["weight"] for item in cluster["items"]], dtype=np.float32)
                cluster["box"] = np.average(boxes, axis=0, weights=weights).tolist()
                placed = True
                break
        if not placed:
            clusters.append({"box": cand["box"], "items": [cand]})
    for cluster in clusters:
        cluster["score"] = sum(item["score"] * item["weight"] for item in cluster["items"])
        cluster["models"] = "+".join(sorted({item["model"] for item in cluster["items"]}))
    return sorted(clusters, key=lambda c: c["score"], reverse=True)


def draw(image: np.ndarray, detections: list[dict], title: str) -> np.ndarray:
    out = image.copy()
    colors = [(0, 64, 255), (0, 220, 0), (255, 128, 0), (255, 0, 220), (0, 220, 255), (180, 80, 255)]
    for idx, det in enumerate(detections, 1):
        x0, y0, x1, y1 = [int(round(v)) for v in det["box"]]
        color = colors[(idx - 1) % len(colors)]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 5)
        label = f"{idx}:ref {det['score']:.2f}"
        cv2.putText(out, label, (x0, max(30, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(out, f"{title} count={len(detections)}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-count", type=int, default=5)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for image_path in sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        candidates = []
        for model, thr, weight in MODEL_CHOICES:
            csv_path = results_dir / model / thr / f"{image_path.stem}_detections.csv"
            candidates.extend(read_detections(csv_path, model, weight))
        clusters = cluster_candidates(candidates)
        selected = clusters[: args.target_count]
        selected = sorted(selected, key=lambda d: (d["box"][0] + d["box"][2]) * 0.5)

        image = imread(image_path)
        if image is None:
            continue
        annotated = draw(image, selected, "codex_reference")
        imwrite(output_dir / f"{image_path.stem}_bbox.png", annotated)
        with (output_dir / f"{image_path.stem}_detections.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "label", "score", "x0", "y0", "x1", "y1", "source_models"])
            for idx, det in enumerate(selected, 1):
                writer.writerow([idx, "person", f"{det['score']:.6f}", *[f"{v:.2f}" for v in det["box"]], det["models"]])
        summary.append([image_path.name, len(selected)])

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "count"])
        writer.writerows(summary)


if __name__ == "__main__":
    main()
