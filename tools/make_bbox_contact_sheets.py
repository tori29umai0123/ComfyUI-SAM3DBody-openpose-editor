from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def imread(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def put_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir)
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for thr_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("thr_")):
            tiles = []
            for img_path in sorted(thr_dir.glob("*_bbox.png")):
                img = imread(img_path)
                if img is None:
                    continue
                scale = 360 / img.shape[1]
                tile = cv2.resize(img, (360, max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
                tiles.append(put_label(tile, img_path.name.replace("_bbox.png", "")))
            if not tiles:
                continue
            max_h = max(t.shape[0] for t in tiles)
            padded = []
            for t in tiles:
                if t.shape[0] < max_h:
                    pad = np.zeros((max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8)
                    t = np.vstack([t, pad])
                padded.append(t)
            sheet = np.hstack(padded)
            out_path = root / f"contact_{model_dir.name}_{thr_dir.name}.png"
            ok, enc = cv2.imencode(".png", sheet)
            if ok:
                enc.tofile(str(out_path))


if __name__ == "__main__":
    main()
